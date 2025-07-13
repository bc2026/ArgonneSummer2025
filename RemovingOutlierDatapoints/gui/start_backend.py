#!/usr/bin/env python3
"""
Backend Startup Script
Simple script to start the Flask backend server
"""

import subprocess
import sys
import os
import time

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'flask', 'flask-cors', 'pandas', 'numpy', 'scipy', 'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        print("   or")
        print("   pip install -r backend_requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def start_backend():
    """Start the Flask backend server"""
    if not check_requirements():
        sys.exit(1)
    
    print("🚀 Starting CSV Time Series Analysis Backend...")
    print("📍 Server will be available at: http://127.0.0.1:5000")
    print("🔧 Health check endpoint: http://127.0.0.1:5000/health")
    print("⏹️  Press Ctrl+C to stop the server\n")
    
    try:
        # Start the Flask server
        subprocess.run([sys.executable, 'backend.py'], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting backend: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ backend.py not found in current directory")
        sys.exit(1)

if __name__ == '__main__':
    start_backend() 