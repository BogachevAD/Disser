"""Собирает автономное Windows-приложение через PyInstaller.

Сценарий можно запускать напрямую кнопкой Run в PyCharm. Он создаёт изолированное
окружение `.build_venv`, устанавливает зависимости, упаковывает точку входа и
README со встроенной справкой в один EXE. Результат помещается в
`dist/IK_Gaussian_Simulator.exe`; Python на целевом компьютере не требуется.
"""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv


APPLICATION_NAME = "IK_Gaussian_Simulator"
ENTRY_POINT = "run_gauss_simulator.py"
BUILD_ENVIRONMENT = ".build_venv"
BUNDLED_DATA_FILES = ("README.md",)


def run_command(command, working_directory, environment=None):
    """Запускает внешнюю command и останавливает сборку при ошибке.

    command — список аргументов без shell-интерпретации; working_directory —
    корень проекта; environment при необходимости заменяет окружение процесса.
    """
    print("\n>", subprocess.list2cmdline([str(item) for item in command]))
    subprocess.run(command, cwd=working_directory, check=True, env=environment)


def ensure_application_is_closed(output_path):
    """Проверяет, доступен ли прежний EXE для замены PyInstaller.

    output_path — ожидаемый файл сборки. На Windows попытка открыть запущенный
    EXE для записи завершается PermissionError; данные при этой проверке не
    изменяются. Это надёжнее tasklist, доступ к которому может быть запрещён.
    """
    if os.name != "nt" or not output_path.is_file():
        return
    process_name = f"{APPLICATION_NAME}.exe"
    try:
        with output_path.open("r+b"):
            pass
    except PermissionError:
        raise RuntimeError(
            f"Закройте {process_name} перед пересборкой: Windows не позволяет "
            "перезаписать запущенный EXE. Затем снова нажмите Run в PyCharm."
        ) from None


def isolated_build_environment(environment_python):
    """Возвращает окружение с безопасным PATH для анализа DLL PyInstaller.

    environment_python задаёт каталоги виртуального окружения. Из PATH удаляются
    сторонние программы, чьи одноимённые DLL могут случайно попасть в EXE.
    """
    environment = os.environ.copy()
    windows_root = Path(environment.get("SystemRoot", r"C:\Windows"))
    python_scripts = environment_python.parent
    python_root = python_scripts.parent
    environment["PATH"] = os.pathsep.join(
        str(path)
        for path in (
            python_scripts,
            python_root,
            windows_root / "System32",
            windows_root,
            windows_root / "System32" / "Wbem",
        )
    )
    return environment


def build_environment_python(project_root, recreate=False):
    """Создаёт или переиспользует отдельное Python-окружение сборки.

    project_root определяет `.build_venv`; recreate удаляет только это окружение
    и создаёт его заново. Возвращается путь к `Scripts/python.exe`.
    """
    environment_path = project_root / BUILD_ENVIRONMENT
    if recreate and environment_path.exists():
        shutil.rmtree(environment_path)
    environment_python = environment_path / "Scripts" / "python.exe"
    if not environment_python.exists():
        print(f"Создание изолированного окружения: {environment_path}")
        venv.EnvBuilder(with_pip=True, clear=False).create(environment_path)
    return environment_python


def install_dependencies(environment_python, project_root):
    """Устанавливает зависимости приложения и PyInstaller.

    environment_python — интерпретатор `.build_venv`; requirements.txt содержит
    runtime-библиотеки, requirements-build.txt — инструменты упаковки.
    """
    run_command(
        [
            environment_python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-r",
            project_root / "requirements.txt",
            "-r",
            project_root / "requirements-build.txt",
        ],
        project_root,
    )


def pyinstaller_command(environment_python, project_root, one_file=True):
    """Формирует аргументы PyInstaller для GUI-приложения.

    environment_python и project_root задают инструмент и пути; one_file
    переключает единый EXE или диагностическую папку. QtAgg указывается явно,
    а стандартный hook Matplotlib добавляет только необходимые данные backend.
    """
    build_root = project_root / "build"
    command = [
        environment_python,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",
        "--name",
        APPLICATION_NAME,
        "--distpath",
        project_root / "dist",
        "--workpath",
        build_root / "pyinstaller-work",
        "--specpath",
        build_root,
        "--paths",
        project_root,
        "--hidden-import",
        "matplotlib.backends.backend_qtagg",
    ]

    # README является ресурсом приложения: MethodHelpDialog читает из него
    # отмеченный раздел как в PyCharm, так и после распаковки one-file EXE.
    for relative_path in BUNDLED_DATA_FILES:
        source_path = project_root / relative_path
        command.extend(["--add-data", f"{source_path}{os.pathsep}."])
    command.append("--onefile" if one_file else "--onedir")

    # Иконка применяется автоматически, если пользователь добавит assets/app.ico.
    icon_path = project_root / "assets" / "app.ico"
    if icon_path.exists():
        command.extend(["--icon", icon_path])
    command.append(project_root / ENTRY_POINT)
    return command


def executable_path(project_root, one_file=True):
    """Возвращает ожидаемый путь результата для выбранного режима.

    В one-file это один EXE в dist; в onedir — EXE внутри одноимённой папки.
    """
    if one_file:
        return project_root / "dist" / f"{APPLICATION_NAME}.exe"
    return project_root / "dist" / APPLICATION_NAME / f"{APPLICATION_NAME}.exe"


def sha256_file(file_path):
    """Вычисляет SHA-256 собранного файла порциями по 1 MiB.

    file_path — путь EXE; возвращаемая hex-строка позволяет проверить, что файл
    не изменился при копировании на другой компьютер.
    """
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_executable(project_root, recreate_environment=False, skip_install=False, one_file=True):
    """Выполняет полную воспроизводимую сборку приложения.

    project_root — каталог исходников; два флага управляют окружением и pip;
    one_file выбирает формат. Возвращается проверенный путь готового EXE.
    """
    ensure_application_is_closed(executable_path(project_root, one_file))
    missing_resources = [
        project_root / relative_path
        for relative_path in BUNDLED_DATA_FILES
        if not (project_root / relative_path).is_file()
    ]
    if missing_resources:
        missing_text = ", ".join(str(path) for path in missing_resources)
        raise FileNotFoundError(f"Не найдены ресурсы для встроенной справки: {missing_text}")
    (project_root / "build").mkdir(exist_ok=True)
    environment_python = build_environment_python(project_root, recreate_environment)
    if not skip_install:
        install_dependencies(environment_python, project_root)
    run_command(
        pyinstaller_command(environment_python, project_root, one_file),
        project_root,
        environment=isolated_build_environment(environment_python),
    )
    result = executable_path(project_root, one_file)
    if not result.is_file() or result.stat().st_size == 0:
        raise RuntimeError(f"PyInstaller завершился без ожидаемого файла: {result}")
    return result


def build_argument_parser():
    """Создаёт разбор параметров сценария сборки.

    Пользователь может пересоздать окружение, пропустить pip или собрать папку
    onedir для диагностики вместо единственного переносимого EXE.
    """
    parser = argparse.ArgumentParser(description="Сборка IK Gaussian Simulator в Windows EXE")
    parser.add_argument("--recreate-env", action="store_true", help="Пересоздать .build_venv")
    parser.add_argument(
        "--skip-install", "--quick", dest="skip_install", action="store_true",
        help="Быстрая повторная сборка: не запускать pip install",
    )
    parser.add_argument("--onedir", action="store_true", help="Собрать папку вместо одного EXE")
    return parser


def main(argv=None):
    """Проверяет платформу, запускает сборку и печатает результат.

    argv — необязательный список аргументов. При успехе выводятся абсолютный
    путь, размер и SHA-256 автономного приложения.
    """
    if os.name != "nt":
        raise SystemExit("Windows EXE необходимо собирать в Windows: PyInstaller не является кросс-компилятором.")
    if sys.version_info < (3, 11):
        raise SystemExit("Для сборки требуется Python 3.11 или новее.")

    arguments = build_argument_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parent
    try:
        result = build_executable(
            project_root,
            recreate_environment=arguments.recreate_env,
            skip_install=arguments.skip_install,
            one_file=not arguments.onedir,
        )
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"\nСборка остановлена: {error}") from None
    size_megabytes = result.stat().st_size / (1024 * 1024)
    print("\nСборка успешно завершена.")
    print(f"Файл: {result}")
    print(f"Размер: {size_megabytes:.1f} MiB")
    print(f"SHA-256: {sha256_file(result)}")
    print("Повторная сборка из PyCharm: запустите build_exe.py с параметром --quick.")


if __name__ == "__main__":
    # Прямой запуск выполняет сборку; импорт позволяет тестировать отдельные шаги.
    main()
