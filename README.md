# Xbox Console Automation

A Python-driven automation framework to control Xbox (and other) consoles using an Arduino Leonardo emulating a controller via GIMX.

---

## 1. System Architecture

```
   YOUR PYTHON SCRIPT
          │
          ▼
   gimx.exe (CLIENT)       "press A"
          │
          │ UDP network message (127.0.0.1:51914)
          ▼
   gimx.exe (SERVER)       owns the hardware
          │
          │ USB cable
          ▼
   FTDI UART adapter       USB <-> serial converter (e.g. COM8)
          │
          │ TX / RX wires
          ▼
   ARDUINO LEONARDO        flashed with GIMX firmware
          │                (console sees this as a controller)
          │ USB cable
          ▼
   XBOX CONSOLE
```

- **Arduino Leonardo**: Special because its ATmega32U4 chip features built-in USB communication, allowing it to pretend to be a USB controller.
- **GIMX**: Translates PC events into controller-compatible USB packets and provides the emulation firmware.

---

## 2. Directory Layout

```
xboxArudino/
├── README.md                      <- Entry point (you are here)
├── docs/                          <- Comprehensive guides
│   ├── 00-README-START-HERE.md    <- In-depth introduction
│   ├── 01-hardware-setup.md       <- Wiring diagrams & hardware list
│   ├── 02-leonardo-flash-docs.md  <- How to flash Leonardo firmware
│   ├── 03-console-connector-docs.md <- Troubleshooting connection
│   ├── 04-buttons-docs.md         <- Mapping friendy names to GIMX
│   ├── 05-test-controller-docs.md <- CLI/library usage guide
│   ├── 06-troubleshooting.md      <- Solution database for common errors
│   ├── 07-lessons-learned.md      <- Development history and post-mortems
│   ├── 08-roadmap-agentic-framework.md <- Next phase: vision-based loop
│   └── 09-gimx-session-docs.md    <- GIMX session manager details
└── Xbox-Automation-Python/
    ├── config/
    │   └── controls.yaml          <- Global single source of truth for controls
    ├── console-connector/
    │   └── console_connector.py   <- Port scan & connection diagnosis utility
    ├── flash-leonardo/
    │   └── flash_leonardo.ps1     <- Flashing automation script
    ├── gimx-session/
    │   ├── gimx_session.py        <- Session lifecycle manager (spawns GIMX)
    │   └── _selftest.py           <- Offline verification for session config
    └── test-controller/
        ├── test_controller.py     <- Main control CLI and Python library
        └── _selftest.py           <- Offline verification for control config
```

---

## 3. Quick Start

### Step 1: Install Dependencies
```bash
pip install -r Xbox-Automation-Python/requirements.txt
```

### Step 2: Start GIMX Server (Terminal 1)
To reserve the serial port and listen for network commands:
```bash
python Xbox-Automation-Python/gimx-session/gimx_session.py start
```
> [!IMPORTANT]
> **Action Required**: You must hold the physical controller's **GUIDE** (Xbox logo) button for **2 seconds** when prompted to authenticate the GIMX session. Skipping this step means GIMX will swallow inputs and they will not reach the console.

### Step 3: Check Connection (Terminal 2)
```bash
python Xbox-Automation-Python/test-controller/test_controller.py --check
```

### Step 4: Run Automation Commands
```bash
# List all mapped buttons, triggers, sticks, and macros
python Xbox-Automation-Python/test-controller/test_controller.py --list

# Send button presses (supports repeats)
python Xbox-Automation-Python/test-controller/test_controller.py press down*3 a

# Run a macro defined in controls.yaml
python Xbox-Automation-Python/test-controller/test_controller.py macro nav_test
```

---

## 4. Safety Warning

> [!WARNING]
> **Do not run GIMX as administrator unless absolutely necessary.**
> If elevated, `gimx.exe` takes realtime CPU priority and grabs mouse/keyboard inputs, which can freeze Windows entirely. 
> The [gimx_session.py](file:///c:/Users/sinha/Desktop/InstrumentsStore/xboxArudino/Xbox-Automation-Python/gimx-session/gimx_session.py) module automatically implements freeze prevention by launching GIMX with `--nograb` and pinning its process priority class back to `Normal`.

---

## 5. Verification

To check code configuration and internal documentation health offline:
```bash
# Verify all internal documentation links resolve correctly
python docs/_verify_docs.py

# Run offline checks for session configuration
python Xbox-Automation-Python/gimx-session/_selftest.py

# Run offline checks for controller mappings and alias resolution
python Xbox-Automation-Python/test-controller/_selftest.py
```
