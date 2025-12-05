#!/usr/bin/env python3
"""
Comprehensive Setup Script for Video-Audio-Face-Emotion-Recognition
Works on Windows, macOS, and Linux
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")

def print_step(text):
    print(f"{Colors.OKBLUE}📋 {text}{Colors.ENDC}")

def run_command(cmd, check=True, shell=False):
    """Run a command and return success status"""
    try:
        if isinstance(cmd, str):
            cmd = cmd.split()
        result = subprocess.run(cmd, check=check, shell=shell, 
                              capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    except FileNotFoundError:
        return False, "Command not found"

def check_python_version():
    """Check if Python version is 3.9 or higher"""
    print_step("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print_error(f"Python 3.9+ required. Found Python {version.major}.{version.minor}")
        return False
    print_success(f"Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def detect_os():
    """Detect operating system"""
    system = platform.system().lower()
    if system == 'windows':
        return 'windows'
    elif system == 'darwin':
        return 'macos'
    elif system == 'linux':
        return 'linux'
    else:
        return 'unknown'

def check_command_exists(cmd):
    """Check if a command exists in PATH"""
    if isinstance(cmd, str):
        cmd = [cmd]
    success, _ = run_command(['which' if os.name != 'nt' else 'where', cmd[0]], check=False)
    return success

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    print_step("Checking for FFmpeg...")
    success, output = run_command(['ffmpeg', '-version'], check=False)
    if success:
        version_line = output.split('\n')[0] if output else "FFmpeg installed"
        print_success(f"FFmpeg found: {version_line}")
        return True
    else:
        print_warning("FFmpeg not found")
        return False

def check_git_lfs():
    """Check if Git LFS is installed"""
    print_step("Checking for Git LFS...")
    success, output = run_command(['git', 'lfs', 'version'], check=False)
    if success:
        version_line = output.split('\n')[0] if output else "Git LFS installed"
        print_success(f"Git LFS found: {version_line}")
        return True
    else:
        print_warning("Git LFS not found")
        return False

def install_ffmpeg(os_type):
    """Provide instructions to install FFmpeg"""
    print_warning("FFmpeg is required for video processing")
    print_info("Please install FFmpeg using one of the following methods:\n")
    
    if os_type == 'macos':
        print("  Option 1 (Recommended):")
        print("    brew install ffmpeg\n")
        print("  Option 2: Download from https://ffmpeg.org/download.html\n")
    elif os_type == 'linux':
        print("  Option 1 (Ubuntu/Debian):")
        print("    sudo apt-get update")
        print("    sudo apt-get install ffmpeg\n")
        print("  Option 2 (RHEL/CentOS):")
        print("    sudo yum install ffmpeg\n")
        print("  Option 3: Download from https://ffmpeg.org/download.html\n")
    elif os_type == 'windows':
        print("  Option 1 (Chocolatey):")
        print("    choco install ffmpeg\n")
        print("  Option 2: Download from https://ffmpeg.org/download.html")
        print("    Extract and add to PATH\n")
    
    response = input("Have you installed FFmpeg? (y/n): ").strip().lower()
    return response == 'y'

def install_git_lfs(os_type):
    """Provide instructions to install Git LFS"""
    print_warning("Git LFS is required to download the combined model file")
    print_info("Please install Git LFS using one of the following methods:\n")
    
    if os_type == 'macos':
        print("  Option 1 (Recommended):")
        print("    brew install git-lfs\n")
        print("  Option 2: Download from https://git-lfs.github.com/\n")
    elif os_type == 'linux':
        print("  Option 1 (Ubuntu/Debian):")
        print("    sudo apt-get install git-lfs\n")
        print("  Option 2: Download from https://git-lfs.github.com/\n")
    elif os_type == 'windows':
        print("  Download installer from: https://git-lfs.github.com/\n")
    
    response = input("Have you installed Git LFS? (y/n): ").strip().lower()
    return response == 'y'

def create_venv():
    """Create virtual environment"""
    print_step("Creating virtual environment...")
    venv_path = Path('venv')
    
    if venv_path.exists():
        print_warning("Virtual environment already exists")
        response = input("Recreate virtual environment? (y/n): ").strip().lower()
        if response == 'y':
            shutil.rmtree(venv_path)
        else:
            print_success("Using existing virtual environment")
            return True
    
    try:
        subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
        print_success("Virtual environment created")
        return True
    except subprocess.CalledProcessError:
        print_error("Failed to create virtual environment")
        return False

def get_pip_command():
    """Get the correct pip command based on OS"""
    if os.name == 'nt':  # Windows
        return str(Path('venv') / 'Scripts' / 'pip')
    else:  # macOS/Linux
        return str(Path('venv') / 'bin' / 'pip')

def get_python_command():
    """Get the correct python command based on OS"""
    if os.name == 'nt':  # Windows
        return str(Path('venv') / 'Scripts' / 'python')
    else:  # macOS/Linux
        return str(Path('venv') / 'bin' / 'python')

def upgrade_pip():
    """Upgrade pip to latest version"""
    print_step("Upgrading pip...")
    pip_cmd = get_pip_command()
    success, _ = run_command([pip_cmd, 'install', '--upgrade', 'pip'], check=False)
    if success:
        print_success("pip upgraded")
    else:
        print_warning("Could not upgrade pip (continuing anyway)")
    return True

def install_pytorch():
    """Install PyTorch"""
    print_step("Installing PyTorch...")
    pip_cmd = get_pip_command()
    
    print_info("Installing PyTorch (CPU version)...")
    print_info("For GPU support, you can install CUDA version later")
    
    success, _ = run_command([pip_cmd, 'install', 'torch', 'torchvision', 'torchaudio'], check=False)
    if success:
        print_success("PyTorch installed")
        return True
    else:
        print_error("Failed to install PyTorch")
        return False

def install_requirements():
    """Install Python requirements"""
    print_step("Installing Python dependencies...")
    pip_cmd = get_pip_command()
    
    if not Path('requirements.txt').exists():
        print_error("requirements.txt not found")
        return False
    
    success, output = run_command([pip_cmd, 'install', '-r', 'requirements.txt'], check=False)
    if success:
        print_success("Python dependencies installed")
        return True
    else:
        print_error("Failed to install some dependencies")
        print_info("You may need to install them manually")
        return False

def install_spacy_model():
    """Download spaCy language model"""
    print_step("Downloading spaCy language model...")
    python_cmd = get_python_command()
    
    success, _ = run_command([python_cmd, '-m', 'spacy', 'download', 'en_core_web_lg'], check=False)
    if success:
        print_success("spaCy model downloaded")
        return True
    else:
        print_warning("Could not download spaCy model (you can do this manually later)")
        print_info("Run: python -m spacy download en_core_web_lg")
        return False

def clone_pytorch_utils():
    """Clone pytorch_utils submodule"""
    print_step("Setting up pytorch_utils submodule...")
    pytorch_utils_path = Path('source') / 'pytorch_utils'
    
    if pytorch_utils_path.exists() and (pytorch_utils_path / '.git').exists():
        print_success("pytorch_utils already exists")
        # Checkout correct version
        os.chdir(pytorch_utils_path)
        run_command(['git', 'checkout', 'v1.0.3'], check=False)
        os.chdir('..')
        os.chdir('..')
        return True
    
    if not pytorch_utils_path.exists():
        pytorch_utils_path.parent.mkdir(parents=True, exist_ok=True)
    
    print_info("Cloning pytorch_utils repository...")
    success, _ = run_command([
        'git', 'clone', 
        'https://github.com/rishiswethan/pytorch_utils.git',
        str(pytorch_utils_path)
    ], check=False)
    
    if success:
        os.chdir(pytorch_utils_path)
        run_command(['git', 'checkout', 'v1.0.3'], check=False)
        os.chdir('..')
        os.chdir('..')
        print_success("pytorch_utils cloned and checked out v1.0.3")
        return True
    else:
        print_error("Failed to clone pytorch_utils")
        print_info("You can clone it manually:")
        print_info("  git clone https://github.com/rishiswethan/pytorch_utils.git source/pytorch_utils")
        print_info("  cd source/pytorch_utils && git checkout v1.0.3")
        return False

def run_setup_py():
    """Run setup.py to create folders"""
    print_step("Creating project folders...")
    python_cmd = get_python_command()
    
    success, _ = run_command([python_cmd, 'setup.py'], check=False)
    if success:
        print_success("Project folders created")
        return True
    else:
        print_warning("Could not run setup.py (you can run it manually later)")
        return False

def check_model_files():
    """Check if model files exist"""
    print_step("Checking for model files...")
    models = {
        'Audio': Path('models') / 'audio' / 'audio_model.pth',
        'Face': Path('models') / 'face' / 'face_model.pth',
        'Combined': Path('models') / 'audio_face_combined' / 'audio_face_combined_model.pth'
    }
    
    all_present = True
    for name, path in models.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > 1:  # Model file should be > 1MB
                print_success(f"{name} model found ({size_mb:.1f} MB)")
            else:
                print_warning(f"{name} model file is too small (likely Git LFS pointer)")
                all_present = False
        else:
            print_warning(f"{name} model not found")
            all_present = False
    
    if not all_present:
        print_info("\nIf models are missing, you may need to:")
        print_info("  1. Install Git LFS: git lfs install")
        print_info("  2. Pull model files: git lfs pull")
        print_info("  3. Or download models manually from the repository")
    
    return all_present

def download_combined_model():
    """Try to download combined model using Git LFS"""
    print_step("Attempting to download combined model...")
    
    model_path = Path('models') / 'audio_face_combined' / 'audio_face_combined_model.pth'
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        if size_mb > 100:  # Actual model should be > 100MB
            print_success("Combined model already present")
            return True
    
    # Check if we're in a git repository
    if Path('.git').exists():
        print_info("Attempting to pull model with Git LFS...")
        success, _ = run_command(['git', 'lfs', 'pull', str(model_path)], check=False)
        if success:
            if model_path.exists() and model_path.stat().st_size > 100 * 1024 * 1024:
                print_success("Combined model downloaded")
                return True
    
    print_warning("Could not download combined model automatically")
    print_info("You can download it manually or use Git LFS later")
    return False

def print_activation_instructions(os_type):
    """Print instructions to activate virtual environment"""
    print_header("Setup Complete!")
    print_success("Project setup completed successfully!\n")
    
    print_info("To activate the virtual environment:\n")
    if os_type == 'windows':
        print("  venv\\Scripts\\activate\n")
    else:
        print("  source venv/bin/activate\n")
    
    print_info("To run the Streamlit UI:\n")
    python_cmd = get_python_command()
    print(f"  {python_cmd} -m streamlit run app.py\n")
    
    print_info("Or use the run script:\n")
    if os_type == 'windows':
        print("  run_ui.bat\n")
    else:
        print("  ./run_ui.sh\n")
    
    print_info("To run the command-line interface:\n")
    print(f"  {python_cmd} run.py\n")

def main():
    """Main setup function"""
    print_header("Video-Audio-Face-Emotion-Recognition Setup")
    
    # Change to script directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Detect OS
    os_type = detect_os()
    print_info(f"Detected OS: {os_type.title()}\n")
    
    # Check system dependencies
    ffmpeg_installed = check_ffmpeg()
    git_lfs_installed = check_git_lfs()
    
    if not ffmpeg_installed:
        if not install_ffmpeg(os_type):
            print_warning("Continuing without FFmpeg (video processing may fail)")
    
    if not git_lfs_installed:
        if not install_git_lfs(os_type):
            print_warning("Continuing without Git LFS (combined model may not download)")
    
    # Create virtual environment
    if not create_venv():
        print_error("Failed to create virtual environment")
        sys.exit(1)
    
    # Upgrade pip
    upgrade_pip()
    
    # Install PyTorch
    if not install_pytorch():
        print_error("Failed to install PyTorch")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print_warning("Some dependencies may not have installed correctly")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            sys.exit(1)
    
    # Install spaCy model
    install_spacy_model()
    
    # Clone pytorch_utils
    clone_pytorch_utils()
    
    # Run setup.py
    run_setup_py()
    
    # Check model files
    check_model_files()
    
    # Try to download combined model
    if git_lfs_installed:
        download_combined_model()
    
    # Print completion message
    print_activation_instructions(os_type)
    
    print_header("Setup Finished!")
    print_info("You're ready to use the Emotion Recognition System!")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

