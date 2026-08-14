"""
probe_link.py - Diagnose the PC -> UART -> Leonardo link.

console_nav.py writes blindly, so "nothing happened" gives us no information.
This script instead sends the GIMX "get adapter type" query and *waits for a
reply*, trying several baud rates. A reply proves the Leonardo is listening on
the UART; silence everywhere means the problem is the wiring / firmware / baud,
not the button data.

    python probe_link.py
    python probe_link.py --port COM8
"""

import argparse
import sys
import time

try:
    import serial
except ImportError:
    sys.exit("pyserial is required:  pip install pyserial")

BYTE_TYPE   = 0x00      # query: what adapter are you?
BYTE_STATUS = 0x01
BYTE_START  = 0x02
BYTE_RESET  = 0x04

# GIMX firmware has shipped at various UART speeds; try the plausible ones.
BAUDS = [500000, 2000000, 1000000, 250000, 115200, 57600, 9600]


def probe(port, baud, verbose=True):
    """Send a few queries at this baud and report anything that comes back."""
    try:
        ser = serial.Serial(port=port, baudrate=baud, timeout=0.6,
                            write_timeout=2)
    except serial.SerialException as e:
        print(f"  {baud:>8} : cannot open port -> {e}")
        return None

    try:
        time.sleep(0.25)
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        replies = []
        for ptype, label in ((BYTE_TYPE, "TYPE"), (BYTE_STATUS, "STATUS")):
            ser.write(bytes([ptype, 0x00]))
            ser.flush()
            time.sleep(0.25)
            data = ser.read(64)
            if data:
                replies.append((label, data))

        if replies:
            print(f"  {baud:>8} : REPLY RECEIVED")
            for label, data in replies:
                print(f"             {label}: {data.hex(' ')}")
            return baud
        print(f"  {baud:>8} : silence")
        return None
    finally:
        ser.close()
        time.sleep(0.15)


def loopback_hint(port):
    """Detect a TX->RX loopback, which would echo our own bytes back."""
    print("\n--- Loopback check (are we just hearing ourselves?) ---")
    try:
        ser = serial.Serial(port=port, baudrate=115200, timeout=0.5)
    except serial.SerialException as e:
        print(f"  cannot open: {e}")
        return
    try:
        probe_bytes = b"\xA5\x5A\xA5\x5A"
        ser.reset_input_buffer()
        ser.write(probe_bytes)
        ser.flush()
        time.sleep(0.3)
        back = ser.read(16)
        if back == probe_bytes:
            print("  TX is wired straight to RX (loopback) - the Leonardo is")
            print("  NOT actually receiving. Check your TX/RX wiring.")
        elif back:
            print(f"  got back: {back.hex(' ')} (something is responding)")
        else:
            print("  no echo (normal if TX/RX are correctly crossed to the board)")
    finally:
        ser.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM8")
    args = ap.parse_args()

    print(f"Probing {args.port} for a GIMX adapter that talks back...\n")
    print("--- Baud rate sweep ---")
    found = [b for b in BAUDS if probe(args.port, b)]

    print("\n=========== RESULT ===========")
    if found:
        print(f"Adapter replied at: {found}")
        print(f"Use it:  python console_nav.py --baud {found[0]} --test")
    else:
        print("No reply at ANY baud rate.")
        print("The Leonardo is not talking back over this UART. Likely causes:")
        print("  1. TX/RX not crossed: FTDI TX -> Leonardo RX(D0),")
        print("     FTDI RX -> Leonardo TX(D1), and GND <-> GND.")
        print("  2. GND not shared between the FTDI and the Leonardo.")
        print("  3. The EMUXONE firmware only accepts data from gimx.exe's own")
        print("     handshake, or expects a different UART speed.")
        print("  4. The Leonardo needs to be powered/enumerated by the console.")
        loopback_hint(args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
