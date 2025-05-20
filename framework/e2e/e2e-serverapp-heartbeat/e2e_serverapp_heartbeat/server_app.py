import time

import flwr as fl

app = fl.server.ServerApp()


@app.main()
def main(grid, context):
    print("Sleep for 10 seconds")
    time.sleep(10)
    print("Done sleeping")
