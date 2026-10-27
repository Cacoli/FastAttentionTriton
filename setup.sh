#!/usr/bin/env bash
# ==============================================================================
# TritonForge: Automated WSL Environment Setup Script
# ==============================================================================
# Sets up a clean virtual environment and installs PyTorch, OpenAI Triton,
# and Matplotlib inside WSL / Ubuntu.
# ==============================================================================

set -e

# Terminal colors
BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BOLD}${BLUE}========================================================================${NC}"
echo -e "${BOLD}${BLUE}TritonForge: WSL Environment Automated Setup${NC}"
echo -e "${BOLD}${BLUE}========================================================================${NC}"

# ------------------------------------------------------------------------------
# 1. Check Python3 and python3-full / python3-venv Availability
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[1/4] Checking system prerequisites...${NC}"

if ! command -v python3 &>/dev/null; then
    echo -e "${RED}[!] ERROR: python3 is not installed.${NC}"
    echo -e "    Please run:"
    echo -e "        sudo apt update && sudo apt install -y python3 python3-full python3-venv"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1)
echo -e "    ${GREEN}[✓] Found: ${PYTHON_VERSION} at $(which python3)${NC}"

# Check for python3-full or python3-venv package on Debian/Ubuntu systems
if command -v dpkg &>/dev/null; then
    if ! dpkg -s python3-full &>/dev/null && ! dpkg -s python3-venv &>/dev/null; then
        echo -e "${YELLOW}[!] WARNING: python3-full or python3-venv was not detected via dpkg.${NC}"
        echo -e "    If venv creation fails below, install it via:"
        echo -e "        sudo apt update && sudo apt install -y python3-full python3-venv"
    else
        echo -e "    ${GREEN}[✓] python3-full / python3-venv package is installed.${NC}"
    fi
fi

# ------------------------------------------------------------------------------
# 2. Clean Up Old Virtual Environments
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[2/4] Cleaning up previous virtual environments...${NC}"

if [ -d "venv" ]; then
    echo -e "    [*] Removing existing 'venv' directory..."
    rm -rf venv
    echo -e "    ${GREEN}[✓] Old 'venv' directory cleaned up.${NC}"
else
    echo -e "    ${GREEN}[✓] No existing 'venv' found. Clean slate ready.${NC}"
fi

# ------------------------------------------------------------------------------
# 3. Create Fresh Virtual Environment
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[3/4] Creating new virtual environment (python3 -m venv venv)...${NC}"

python3 -m venv venv

if [ ! -f "venv/bin/python" ]; then
    echo -e "${RED}[!] ERROR: Virtual environment creation failed.${NC}"
    echo -e "    Please install python3-full: sudo apt update && sudo apt install -y python3-full"
    exit 1
fi

echo -e "    ${GREEN}[✓] Virtual environment created successfully at ./venv${NC}"

# ------------------------------------------------------------------------------
# 4. Upgrade pip & Install Core Packages
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[4/4] Upgrading pip and installing required dependencies...${NC}"

echo -e "    [*] Upgrading pip..."
./venv/bin/python -m pip install --upgrade pip

echo -e "    [*] Installing torch, triton, and matplotlib..."
./venv/bin/python -m pip install torch triton matplotlib

echo -e "    ${GREEN}[✓] Dependencies installed successfully.${NC}"

# ------------------------------------------------------------------------------
# 5. Success Banner & Instructions
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}${GREEN}========================================================================${NC}"
echo -e "${BOLD}${GREEN}[+] TritonForge WSL Environment Setup Complete!${NC}"
echo -e "${BOLD}${GREEN}========================================================================${NC}"
echo ""
echo -e "${BOLD}Next steps:${NC}"
echo ""
echo -e "  1. Activate the virtual environment:"
echo -e "     ${YELLOW}source venv/bin/activate${NC}"
echo ""
echo -e "  2. Run the 48-configuration correctness verification suite:"
echo -e "     ${YELLOW}python test_correctness.py${NC}"
echo -e "     # Or for a quick 4-case sanity check:"
echo -e "     ${YELLOW}python test_correctness.py --quick${NC}"
echo ""
echo -e "  3. Run the performance benchmark suite against PyTorch SDPA:"
echo -e "     ${YELLOW}python benchmark.py --seq-lens 512 1024 2048 4096${NC}"
echo -e "     # Or with causal masking:"
echo -e "     ${YELLOW}python benchmark.py --seq-lens 512 1024 2048 4096 --causal${NC}"
echo ""
echo -e "${BOLD}${BLUE}========================================================================${NC}"
