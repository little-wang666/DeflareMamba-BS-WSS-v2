import argparse
import py_compile
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MIN_LINE_COUNTS = {
    'basicsr/archs/wavelet_vssm_modules.py': 100,
    'basicsr/archs/DeflareMamba_arch.py': 500,
    'scripts/test_bs_wss_forward.py': 50,
    'scripts/check_text_format.py': 40,
    'scripts/verify_v2_6.py': 50,
    'options/DeflareMamba_flare7kpp_bs_wss_option.yml': 100,
}

TEXT_FILES = [
    '.gitattributes',
    '.editorconfig',
    'README.md',
    'options/DeflareMamba_flare7kpp_baseline_option.yml',
    *MIN_LINE_COUNTS.keys(),
]

PY_COMPILE_FILES = [
    'basicsr/archs/wavelet_vssm_modules.py',
    'basicsr/archs/DeflareMamba_arch.py',
    'scripts/test_bs_wss_forward.py',
    'scripts/check_text_format.py',
    'scripts/verify_v2_6.py',
]


def read_text_file(relative_name):
    path = ROOT / relative_name
    data = path.read_bytes()
    if b'\x00' in data:
        raise AssertionError(f'{relative_name}: contains NUL bytes')
    if b'\r' in data:
        raise AssertionError(f'{relative_name}: contains CR characters; expected LF-only line endings')
    if b'\n' not in data:
        raise AssertionError(f'{relative_name}: appears to be a single-line file')
    return path, data.decode('utf-8')


def count_lines(text):
    return len(text.splitlines())


def run_line_checks():
    print('line count checks:')
    for relative_name in TEXT_FILES:
        _, text = read_text_file(relative_name)
        lines = count_lines(text)
        minimum = MIN_LINE_COUNTS.get(relative_name)
        if minimum is not None and lines <= minimum:
            raise AssertionError(f'{relative_name}: {lines} lines, expected > {minimum}')
        print(f'  {relative_name}: {lines} lines')


def run_py_compile():
    print('py_compile checks:')
    for relative_name in PY_COMPILE_FILES:
        py_compile.compile(str(ROOT / relative_name), doraise=True)
        print(f'  {relative_name}: ok')


def run_text_format_check():
    print('text format check:')
    subprocess.run([sys.executable, str(ROOT / 'scripts/check_text_format.py')], cwd=ROOT, check=True)


def run_forward_tests():
    print('forward shape checks:')
    for case in ('baseline', 'bs_wss'):
        subprocess.run(
            [sys.executable, str(ROOT / 'scripts/test_bs_wss_forward.py'), '--case', case],
            cwd=ROOT,
            check=True,
        )


def main():
    parser = argparse.ArgumentParser(description='Verify DeflareMamba BS-WSS v2.6 repository health.')
    parser.add_argument('--skip-forward', action='store_true', help='Skip torch/mamba forward tests.')
    args = parser.parse_args()

    run_line_checks()
    run_py_compile()
    run_text_format_check()
    if args.skip_forward:
        print('forward shape checks: skipped by --skip-forward')
    else:
        run_forward_tests()


if __name__ == '__main__':
    main()
