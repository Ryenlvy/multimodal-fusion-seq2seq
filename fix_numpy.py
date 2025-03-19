import subprocess
import sys

def downgrade_numpy():
    """Downgrade NumPy to version 1.x"""
    print("Downgrading NumPy to version 1.x...")

    try:
        # Uninstall the current NumPy
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "numpy"])

        # Install NumPy 1.x
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy<2.0.0"])

        # Verify installation
        subprocess.check_call([sys.executable, "-c", "import numpy; print(f'NumPy version: {numpy.__version__}')"])

        print("NumPy downgrade successful!")
    except Exception as e:
        print(f"Downgrade failed: {e}")
        print("Please try running manually: pip uninstall -y numpy && pip install numpy<2.0.0")

if __name__ == "__main__":
    downgrade_numpy()
