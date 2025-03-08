import asyncio
import websockets
from threading import Thread
import json
class Remote():
    IDLE = "idle"
    STANDING_UP = "standing_up"
    STANDING = "standing"
    COLLAPSE = "collapse"
    LEARN = "learn"
    PAUSE = "pause"
    IMU_CALIBRATION = "imu_calibration"
    def __init__(self, mode):
        self.mode = mode
        self.connections = set()
        self.loop = asyncio.get_event_loop()
        self.thread = Thread(target=self.loop.run_until_complete, args=(self.main(),))
        self.thread.start()
        print("Remote initialized")
        self.goal_x_velocity = "0"
        self.goal_y_velocity = "0"
        self.goal_yaw_velocity = "0"
        self.state = self.set_state(self.IDLE)

        self.train_goal_x_velocity = "0"
        self.train_goal_y_velocity = "0"
        self.train_goal_yaw_velocity = "0"

    async def handler(self, websocket):
        self.connections.add(websocket)
        if len(self.connections) == 1:
            self.set_state(self.STANDING_UP)
        try:
            while True:
                message = await websocket.recv()
                msg = json.loads(message)
                if msg["state"] == "sending_command" and self.mode == "test":
                    self.goal_x_velocity = msg["command"]["x"]
                    self.goal_y_velocity = msg["command"]["y"]
                    self.goal_yaw_velocity = msg["command"]["yaw"]
                else:
                    if msg["state"] == self.COLLAPSE:
                        self.set_state(self.COLLAPSE)
                    elif msg["state"] == self.STANDING_UP:
                        self.set_state(self.STANDING_UP)
                    elif msg["state"] == self.LEARN:
                        if self.state == self.STANDING:
                            # self.set_state(self.LEARN)
                            self.set_state(self.IMU_CALIBRATION)
                        elif self.state == self.PAUSE:
                            self.set_state(self.LEARN)
                        else:
                            print("Robot is not standing")
                    elif msg["state"] == self.PAUSE:
                        self.set_state(self.PAUSE)
                    else:
                        print("Unknown state")
                        self.set_state(self.COLLAPSE)
                print(f"Received message: {message}")
        except:
            self.connections.remove(websocket)
            if len(self.connections) == 0:
                self.set_state(self.IDLE)
            print("Connection closed")



    async def main(self):
        async with websockets.serve(self.handler, "", 8001):
            await asyncio.Future()  # run forever
    
    def get_state(self):
        return self.state
    
    def set_state(self, state):
        self.state = state
        msg = {
            "state": state,
            "mode": self.mode
        }
        websockets.broadcast(self.connections, json.dumps(msg))
    
    def is_reset(self):
        return self.state == self.COLLAPSE or self.state == self.IDLE
    
    def set_train_goal(self, x, y, yaw):
        self.train_goal_x_velocity = x
        self.train_goal_y_velocity = y
        self.train_goal_yaw_velocity = yaw
        msg = {
            "state": self.state,
            "mode": self.mode,
            "train_goal": {
                "x": round(x, 2),
                "y": round(y, 2),
                "yaw": round(yaw, 2)
            }
        }
        websockets.broadcast(self.connections, json.dumps(msg))