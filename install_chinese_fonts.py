import os
import subprocess
import sys
import platform

def install_chinese_fonts():
    """Install Chinese fonts"""
    system = platform.system()
    
    print(f"Detected operating system: {system}")
    
    if system == "Linux":
        try:
            # Check if the system is Ubuntu/Debian
            if os.path.exists("/etc/debian_version"):
                print("Detected Debian/Ubuntu system, using apt to install Chinese fonts...")
                subprocess.check_call(["sudo", "apt-get", "update"])
                subprocess.check_call(["sudo", "apt-get", "install", "-y", "fonts-wqy-microhei", "fonts-wqy-zenhei"])
            
            # Check if the system is CentOS/RHEL
            elif os.path.exists("/etc/redhat-release"):
                print("Detected CentOS/RHEL system, using yum to install Chinese fonts...")
                subprocess.check_call(["sudo", "yum", "install", "-y", "wqy-microhei-fonts", "wqy-zenhei-fonts"])
            
            # Other Linux distributions
            else:
                print("Unrecognized Linux distribution. Please install Chinese fonts manually.")
                print("Recommended fonts: WenQuanYi Micro Hei, WenQuanYi Zen Hei")
                return False
            
            # Refresh font cache
            subprocess.check_call(["fc-cache", "-fv"])
            print("Chinese font installation completed!")
            return True
            
        except Exception as e:
            print(f"Error installing Chinese fonts: {e}")
            print("Please try installing Chinese fonts manually.")
            return False
    
    elif system == "Windows":
        print("Windows systems usually come with pre-installed Chinese fonts, no additional installation is needed.")
        return True
    
    elif system == "Darwin":  # macOS
        try:
            print("Detected macOS system, using brew to install Chinese fonts...")
            # Check if Homebrew is installed
            try:
                subprocess.check_call(["brew", "--version"])
            except:
                print("Homebrew not detected. Please install Homebrew first: https://brew.sh/")
                return False
            
            # Install Chinese fonts
            subprocess.check_call(["brew", "tap", "homebrew/cask-fonts"])
            subprocess.check_call(["brew", "install", "--cask", "font-wqy-microhei", "font-wqy-zenhei"])
            print("Chinese font installation completed!")
            return True
            
        except Exception as e:
            print(f"Error installing Chinese fonts: {e}")
            print("Please try installing Chinese fonts manually.")
            return False
    
    else:
        print(f"Unsupported operating system: {system}")
        return False

if __name__ == "__main__":
    install_chinese_fonts()
