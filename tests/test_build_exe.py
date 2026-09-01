"""Тесты конфигурации автономной Windows-сборки.

Они не запускают тяжёлый PyInstaller, а проверяют формирование команды и пути,
чтобы изменения сценария не потеряли onefile/windowed или точку входа.
"""

from pathlib import Path
import unittest

from build_exe import APPLICATION_NAME, executable_path, isolated_build_environment, pyinstaller_command


class BuildExeTests(unittest.TestCase):
    """Проверяет чистые функции build_exe без сети и файловой сборки.

    Временные пути намеренно не обязаны существовать: функции только формируют
    аргументы и не запускают внешние процессы.
    """

    def test_onefile_command_is_windowed_and_uses_entry_point(self):
        """Проверяет обязательные ключи переносимого GUI-EXE.

        environment_python и project_root передаются как условные Windows-пути.
        """
        root = Path("C:/project")
        command = pyinstaller_command(Path("C:/venv/Scripts/python.exe"), root, one_file=True)
        self.assertIn("--onefile", command)
        self.assertIn("--windowed", command)
        self.assertNotIn("--collect-all", command)
        self.assertIn("matplotlib.backends.backend_qtagg", command)
        self.assertEqual(command[-1], root / "run_gauss_simulator.py")

    def test_output_path_depends_on_bundle_mode(self):
        """Различает путь единственного EXE и диагностической папки.

        project_root задаёт dist; APPLICATION_NAME определяет имя результата.
        """
        root = Path("C:/project")
        self.assertEqual(executable_path(root, True), root / "dist" / f"{APPLICATION_NAME}.exe")
        self.assertEqual(
            executable_path(root, False),
            root / "dist" / APPLICATION_NAME / f"{APPLICATION_NAME}.exe",
        )

    def test_build_path_excludes_unrelated_native_tools(self):
        """Не допускает захват одноимённых DLL из внешних программ.

        environment_python определяет разрешённые каталоги Python; внешний PATH
        намеренно не должен переходить в процесс анализа PyInstaller.
        """
        python = Path("C:/project/.build_venv/Scripts/python.exe")
        environment = isolated_build_environment(python)
        self.assertIn(str(python.parent), environment["PATH"])
        self.assertNotIn("poppler", environment["PATH"].lower())


if __name__ == "__main__":
    # Прямой запуск выполняет только быстрые unit-тесты, а не реальную сборку.
    unittest.main()
