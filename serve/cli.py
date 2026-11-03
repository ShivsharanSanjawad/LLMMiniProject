import argparse
import sys
import requests

def send_request(url: str, req: str):
    try:
        resp = requests.post(f"{url}/generate", json={"request": req}, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            cmd = data["command"]
            dang = data["dangerous"]
            lat = data["latency_ms"]

            print(f" -> Generated Command: \033[92m{cmd}\033[0m")
            print(f" -> Latency: {lat:.1f} ms")
            if dang:
                print(f" -> \033[91m[DANGER FLAG]: Risky/destructive command pattern detected!\033[0m")
            else:
                print(f" -> [Safety Check]: Passed (Safe)")
        else:
            print(f"Error {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"Connection error: {e}")

def main():
    parser = argparse.ArgumentParser(description="CLI demo for NL2Shell API")
    parser.add_argument("query", nargs="*", help="Optional single English query to translate immediately")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="FastAPI server URL")
    args = parser.parse_args()

    # If query is passed on command line, translate and exit
    if args.query:
        query_str = " ".join(args.query)
        print(f"\n[Query]: {query_str}")
        send_request(args.url, query_str)
        return

    # Interactive mode
    print("==================================================")
    print("      NL2Shell Live Interactive CLI Demo          ")
    print(f"      Connected to: {args.url}                   ")
    print("      Type 'exit' or 'quit' to stop               ")
    print("==================================================")

    while True:
        try:
            req = input("\nEnter English request: ").strip()
            if not req:
                continue
            if req.lower() in ["exit", "quit"]:
                print("Exiting demo.")
                break
            send_request(args.url, req)
        except KeyboardInterrupt:
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()

