import argparse
import py_compile
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

LINE_COUNT_FILES = [
    'basicsr/archs/wavelet_vssm_modules.py',
    'basicsr/archs/DeflareMamba_arch.py',
    'scripts/test_bs_wss_forward.py',
    'scripts/check_text_format.py',
    'scripts/verify_v2_5.py',
    'options/DeflareMamba_flare7kpp_baseline_option.yml',
    'options/DeflareMamba_flare7kpp_bs_wss_option.yml',
    'README.md',
]

PY_COMPILE_FILES = [
    'basicsr/archs/wavelet_vssm_modules.py',
    'basicsr/archs/DeflareMamba_arch.py',
    'scripts/test_bs_wss_forward.py',
    'scripts/check_text_format.py',
    'scripts/verify_v2_5.py',
]


def count_lines(path):
    data = path.read_bytes()
    if b'\x00' in data:
        raise AssertionError(f'{path}: contains NUL bytes')
    if b'\r' in data:
        raise AssertionError(f'{path}: contains CR characters; expected LF-only line endings')
    line_count = data.count(b'\n')
    if line_count == 0:
        raise AssertionError(f'{path}: appears to be a single-line file')
    return line_count + 1


def run_line_checks():
    print('line count checks:')
    for name in LINE_COUNT_FILES:
        path = ROOT / name
        lines = count_lines(path)
        print(f'  {name}: {lines} lines')


def run_py_compile():
    print('py_compile checks:')
    for name in PY_COMPILE_FILES:
        py_compile.compile(str(ROOT / name), doraise=True)
        print(f'  {name}: ok')


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
    parser = argparse.ArgumentParser(description='Verify DeflareMamba BS-WSS v2.5 repository health.')
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
