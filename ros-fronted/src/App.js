import logo from './logo.svg';
import './App.css';
import { useState, useCallback } from 'react';
import useWebSocket, { ReadyState } from 'react-use-websocket';


function App() {
  const [socketUrlDisplay, setSocketUrlDisplay] = useState('ws://192.168.1.27:8001');
  const [socketUrl, setSocketUrl] = useState('ws://192.168.1.27:8001');
  const [mode, setMode] = useState("test");
  const [command, setCommand] = useState({ x: "0", y: "0", yaw: "0" });
  const { sendMessage, lastMessage, readyState } = useWebSocket(socketUrl, { retryOnError: true, shouldReconnect: () => true });

  const connectionStatus = {
    [ReadyState.CONNECTING]: 'Connecting',
    [ReadyState.OPEN]: 'Open',
    [ReadyState.CLOSING]: 'Closing',
    [ReadyState.CLOSED]: 'Closed',
    [ReadyState.UNINSTANTIATED]: 'Uninstantiated',
  }[readyState];

  const defaultMsg = {
    "state": "standing_up"
  }

  const handleClickChangeSocketUrl = () => {
    setSocketUrl(socketUrlDisplay);
  }

  const handleClickChangeSocketUrlLocalhost = () => {
    setSocketUrl('ws://localhost:8001');
    setSocketUrlDisplay('ws://localhost:8001');
  }

  const getMsg = (state, c) => {
    var msg = defaultMsg;
    msg.state = state;
    console.log(c)
    msg.command = c;
    return JSON.stringify(msg);
  }

  const handleClickSendMessage = useCallback((state, c) => sendMessage(getMsg(state, c)), []);


  return (
    <div className="App">
      <header className="App-header">
        <p>
          On-robot deep reinforcement learning for quadrupedal locomotion
        </p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 5, justifyContent: 'center', alignContent: 'center' }}>
          <div style={{ display: 'flex', flexDirection: 'row', gap: 5, justifyContent: 'center', alignContent: 'center' }}>
            <input
              type="text"
              value={socketUrlDisplay}
              onChange={(e) => setSocketUrlDisplay(e.target.value)}
            />
            <button onClick={handleClickChangeSocketUrl}>Connect</button>
            <button onClick={handleClickChangeSocketUrlLocalhost}>Connect Localhost</button>
            <div style={{ color: connectionStatus === "Open" ? "green" : "red" }}>{connectionStatus}</div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'row', gap: 5, justifyContent: 'center', alignContent: 'center' }}>
            <button onClick={() => handleClickSendMessage("standing_up", command)}>standing_up</button>
            <button onClick={() => handleClickSendMessage("learn", command)}>Learn</button>
            <button onClick={() => handleClickSendMessage("collapse", command)} >Collapse</button>
            <button onClick={() => handleClickSendMessage("pause", command)} >Pause</button>
          </div>
          <p>
            {lastMessage ? lastMessage.data : null}
          </p>
        </div>
        <div>
          {(lastMessage?.data ? JSON.parse(lastMessage.data).mode == "test" : false) &&
            <div style={{ display: 'flex', flexDirection: 'row', gap: 5 }}>
              {Object.entries(command).map(([key, value]) => (
                <input
                  style={{ width: 50 }}
                  key={key}
                  type="number"
                  step="0.1"
                  value={value}
                  onChange={(e) => setCommand(prevState => ({ ...prevState, [key]: e.target.value }))}
                />
              ))}
              <button onClick={() => handleClickSendMessage("sending_command", command)}>Send Command</button>
            </div>
          }
        </div>
      </header>
    </div>
  );
}

export default App;
