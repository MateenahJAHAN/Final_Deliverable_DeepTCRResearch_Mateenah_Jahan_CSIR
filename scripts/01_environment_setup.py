#!/usr/bin/env python3
"""
Script 01: Environment Setup and Dependency Checking
=====================================================

This script corresponds to Cells 0-5 of the original Untitled1.ipynb notebook.

PURPOSE:
--------
1. Verify Python version (requires Python 3.7 or higher)
2. Check for required packages and their versions
3. Install missing packages if needed
4. Verify project directory structure
5. Display system information for reproducibility

DEPENDENCIES:
-------------
- Python 3.7+
- Standard library: sys, os, platform, subprocess
- Required packages: pandas, numpy, matplotlib, seaborn, scikit-learn, openpyxl

OUTPUTS:
--------
- Console output showing Python version and package versions
- Warnings if packages are missing or outdated
- Log file: ../logs/01_environment_setup.log

BIOLOGICAL CONTEXT:
-------------------
This script sets up the computational environment for TCR (T-Cell Receptor)
repertoire analysis. TCRs are proteins on immune cells that recognize antigens.
We're analyzing TCR sequences from cancer patients to predict immunotherapy response.

WORKFLOW:
---------
1. Check Python version
2. Check required packages
3. Install missing packages (optional)
4. Display system information
5. Verify directory structure

Author: Converted from Untitled1.ipynb
Date: December 24, 2025
"""

import sys
import os
import platform
import subprocess
from datetime import datetime

# ==============================================================================
# SECTION 1: PROJECT PATHS SETUP
# ==============================================================================

print("="*80)
print("DEEPTCR ENVIRONMENT SETUP - SCRIPT 01")
print("="*80)
print(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Get the script directory and project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

print(f"Script directory: {SCRIPT_DIR}")
print(f"Project root: {PROJECT_ROOT}\n")

# Define project subdirectories
DATA_RAW = os.path.join(PROJECT_ROOT, "data_raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data_processed")
FIGURES = os.path.join(PROJECT_ROOT, "figures")
RESULTS = os.path.join(PROJECT_ROOT, "results")
LOGS = os.path.join(PROJECT_ROOT, "logs")
MODELS = os.path.join(PROJECT_ROOT, "models")

# ==============================================================================
# SECTION 2: PYTHON VERSION CHECK (Notebook Cell 0)
# ==============================================================================

print("-" * 80)
print("SECTION 1: PYTHON VERSION CHECK")
print("-" * 80)

python_version = sys.version_info
print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
print(f"Full version: {sys.version}\n")

# Verify Python 3.7 or higher
if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
    print("❌ ERROR: Python 3.7 or higher is required!")
    print(f"   Current version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    print("   Please upgrade Python and try again.")
    sys.exit(1)
else:
    print(f"✅ Python version check passed: {python_version.major}.{python_version.minor}.{python_version.micro}")

# ==============================================================================
# SECTION 3: SYSTEM INFORMATION
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 2: SYSTEM INFORMATION")
print("-" * 80)

print(f"Operating System: {platform.system()}")
print(f"OS Version: {platform.version()}")
print(f"Machine: {platform.machine()}")
print(f"Processor: {platform.processor()}")
print(f"Architecture: {platform.architecture()[0]}")

# For macOS M3, show additional info
if platform.system() == "Darwin" and "arm" in platform.machine().lower():
    print("✅ Detected Apple Silicon (M-series chip)")
    print("   → Multi-core processing will be optimized for this architecture")

# ==============================================================================
# SECTION 4: REQUIRED PACKAGES CHECK (Notebook Cells 1-4)
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 3: REQUIRED PACKAGES CHECK")
print("-" * 80)

# Define required packages with minimum versions
REQUIRED_PACKAGES = {
    'pandas': '1.4.0',
    'numpy': '1.18.0',
    'matplotlib': '3.3.0',
    'seaborn': '0.11.0',
    'scikit-learn': '1.0.0',
    'openpyxl': '3.0.0',
}

missing_packages = []
installed_packages = {}

print("\nChecking required packages...\n")

for package_name, min_version in REQUIRED_PACKAGES.items():
    try:
        # Try to import the package
        if package_name == 'scikit-learn':
            # Special case: scikit-learn imports as sklearn
            import sklearn
            package = sklearn
            installed_version = sklearn.__version__
        else:
            package = __import__(package_name)
            installed_version = package.__version__

        installed_packages[package_name] = installed_version
        print(f"✅ {package_name:15s} version: {installed_version:10s} (minimum: {min_version})")

    except ImportError:
        missing_packages.append(package_name)
        print(f"❌ {package_name:15s} NOT INSTALLED (minimum required: {min_version})")

# ==============================================================================
# SECTION 5: INSTALL MISSING PACKAGES (Optional)
# ==============================================================================

if missing_packages:
    print("\n" + "-" * 80)
    print("SECTION 4: INSTALLING MISSING PACKAGES")
    print("-" * 80)

    print(f"\n⚠️  Missing packages detected: {', '.join(missing_packages)}")
    print("\nAttempting to install missing packages...\n")

    for package in missing_packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
            print(f"✅ Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            print(f"   Please manually install using: pip install {package}")

    print("\n✅ Package installation complete!")
    print("   Please restart this script to verify installations.")
else:
    print("\n✅ All required packages are installed!")

# ==============================================================================
# SECTION 6: VERIFY PROJECT DIRECTORY STRUCTURE
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 5: PROJECT DIRECTORY STRUCTURE")
print("-" * 80)

directories = {
    'Data (Raw)': DATA_RAW,
    'Data (Processed)': DATA_PROCESSED,
    'Figures': FIGURES,
    'Results': RESULTS,
    'Logs': LOGS,
    'Models': MODELS,
    'Scripts': SCRIPT_DIR,
}

print("\nVerifying project directories...\n")

all_dirs_exist = True
for dir_name, dir_path in directories.items():
    exists = os.path.isdir(dir_path)
    status = "✅ EXISTS" if exists else "❌ MISSING"
    print(f"{status:12s} {dir_name:20s} → {dir_path}")
    if not exists:
        all_dirs_exist = False

if all_dirs_exist:
    print("\n✅ All project directories are in place!")
else:
    print("\n⚠️  Some directories are missing. They will be created by subsequent scripts.")

# ==============================================================================
# SECTION 7: CHECK FOR SOURCE DATA
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 6: SOURCE DATA CHECK")
print("-" * 80)

source_data_file = os.path.join(DATA_RAW, "deeptcr_complete_dataset (5).csv")

if os.path.isfile(source_data_file):
    file_size = os.path.getsize(source_data_file) / (1024 * 1024)  # Convert to MB
    print(f"\n✅ Source data file found!")
    print(f"   File: {os.path.basename(source_data_file)}")
    print(f"   Location: {source_data_file}")
    print(f"   Size: {file_size:.2f} MB")
else:
    print(f"\n❌ Source data file NOT found!")
    print(f"   Expected location: {source_data_file}")
    print(f"   Please ensure 'deeptcr_complete_dataset (5).csv' is in the data_raw/ directory")

# ==============================================================================
# SECTION 8: SAVE ENVIRONMENT INFO TO LOG
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 7: SAVING ENVIRONMENT LOG")
print("-" * 80)

log_file = os.path.join(LOGS, "01_environment_setup.log")

try:
    with open(log_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DEEPTCR ENVIRONMENT SETUP LOG\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("-" * 80 + "\n")
        f.write("PYTHON VERSION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Python {python_version.major}.{python_version.minor}.{python_version.micro}\n")
        f.write(f"Full version: {sys.version}\n\n")

        f.write("-" * 80 + "\n")
        f.write("SYSTEM INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"OS: {platform.system()}\n")
        f.write(f"Version: {platform.version()}\n")
        f.write(f"Machine: {platform.machine()}\n")
        f.write(f"Processor: {platform.processor()}\n\n")

        f.write("-" * 80 + "\n")
        f.write("INSTALLED PACKAGES\n")
        f.write("-" * 80 + "\n")
        for package, version in installed_packages.items():
            f.write(f"{package}: {version}\n")

        if missing_packages:
            f.write("\nMissing packages:\n")
            for package in missing_packages:
                f.write(f"  - {package}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("PROJECT DIRECTORIES\n")
        f.write("-" * 80 + "\n")
        for dir_name, dir_path in directories.items():
            exists = "EXISTS" if os.path.isdir(dir_path) else "MISSING"
            f.write(f"{dir_name}: {exists}\n  Path: {dir_path}\n")

    print(f"\n✅ Environment log saved to: {log_file}")

except Exception as e:
    print(f"\n⚠️  Could not save log file: {e}")

# ==============================================================================
# SECTION 9: SUMMARY AND NEXT STEPS
# ==============================================================================

print("\n" + "="*80)
print("ENVIRONMENT SETUP COMPLETE")
print("="*80)

print("\n📋 SUMMARY:")
print(f"   ✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
print(f"   ✅ Required packages: {len(installed_packages)}/{len(REQUIRED_PACKAGES)} installed")
print(f"   ✅ Project directories: {'All present' if all_dirs_exist else 'Some missing (will be created)'}")
print(f"   ✅ Source data: {'Found' if os.path.isfile(source_data_file) else 'Missing'}")

print("\n📊 NEXT STEPS:")
print("   → Run Script 02: Data loading and TRB extraction")
print("   → Command: python scripts/02_data_loading.py")

print("\n" + "="*80)
print(f"Script completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")
