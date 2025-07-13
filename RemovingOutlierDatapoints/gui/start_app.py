#!/usr/bin/env python3
"""
Complete App Startup Script
Starts both the Python backend and Electron frontend
"""

import subprocess
import sys
import os
import time
import threading
import signal

def check_node_installed():
    """Check if Node.js is installed"""
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Node.js not found")
    print("💡 Please install Node.js from https://nodejs.org")
    return False

def check_npm_deps():
    """Check if npm dependencies are installed"""
    if os.path.exists('node_modules'):
        print("✅ Node.js dependencies found")
        return True
    else:
        print("❌ Node.js dependencies not found")
        print("💡 Run 'npm install' to install dependencies")
        return False

def check_python_deps():
    """Check if Python dependencies are installed"""
    required_packages = ['flask', 'flask_cors', 'pandas', 'numpy']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing Python packages: {', '.join(missing)}")
        print("💡 Run 'pip install -r backend_requirements.txt'")
        return False
    
    print("✅ Python dependencies found")
    return True

def start_backend():
    """Start the Python backend in a separate process"""
    print("🐍 Starting Python backend...")
    try:
        process = subprocess.Popen([sys.executable, 'backend.py'])
        return process
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None

def start_frontend():
    """Start the Electron frontend"""
    print("⚡ Starting Electron frontend...")
    try:
        # Give backend time to start
        time.sleep(3)
        process = subprocess.Popen(['npm', 'start'])
        return process
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        return None

def main():
    print("🚀 CSV Time Series Analysis Tool - Complete Startup")
    print("=" * 50)
    
    # Check prerequisites
    print("\n🔍 Checking prerequisites...")
    
    if not check_node_installed():
        sys.exit(1)
    
    if not check_npm_deps():
        print("\n📦 Installing Node.js dependencies...")
        try:
            subprocess.run(['npm', 'install'], check=True)
            print("✅ Node.js dependencies installed")
        except subprocess.CalledProcessError:
            print("❌ Failed to install Node.js dependencies")
            sys.exit(1)
    
    if not check_python_deps():
        print("\n📦 Installing Python dependencies...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'backend_requirements.txt'], check=True)
            print("✅ Python dependencies installed")
        except subprocess.CalledProcessError:
            print("❌ Failed to install Python dependencies")
            sys.exit(1)
    
    # Start services
    print("\n🚀 Starting services...")
    
    backend_process = start_backend()
    if not backend_process:
        sys.exit(1)
    
    frontend_process = start_frontend()
    if not frontend_process:
        backend_process.terminate()
        sys.exit(1)
    
    print("\n✅ Both services started successfully!")
    print("🌐 Backend: http://127.0.0.1:5000")
    print("🖥️  Frontend: Electron app should open automatically")
    print("\n⏹️  Press Ctrl+C to stop both services")
    
    # Handle shutdown
    def signal_handler(sig, frame):
        print("\n🛑 Shutting down services...")
        try:
            frontend_process.terminate()
            backend_process.terminate()
            
            # Wait for graceful shutdown
            frontend_process.wait(timeout=5)
            backend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("🔨 Force killing processes...")
            frontend_process.kill()
            backend_process.kill()
        
        print("✅ Services stopped")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Wait for both processes
        while True:
            if backend_process.poll() is not None:
                print("❌ Backend process stopped unexpectedly")
                break
            if frontend_process.poll() is not None:
                print("❌ Frontend process stopped")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    
    # Clean up
    try:
        frontend_process.terminate()
        backend_process.terminate()
    except:
        pass

if __name__ == '__main__':
    main() 