import { useState } from 'react'
import MapView from './components/MapView'
import OrderPanel from './components/OrderPanel'
import { useAgentRoute } from './hooks/useWebSocket'

function App() {
  const agentId = 'agent_001'

  // route and alert live HERE — shared between both components
  const { alert } = useAgentRoute(agentId)
  const [route, setRoute] = useState([])

  return (
    <div style={{
      display: 'flex', flexDirection: 'row',
      height: '100vh', width: '100vw',
      margin: 0, padding: 0, overflow: 'hidden'
    }}>
      <div style={{ flex: 1, borderRight: '1px solid #222', overflowY: 'auto' }}>
        <OrderPanel
          agentId={agentId}
          route={route}
          setRoute={setRoute}
        />
      </div>
      <div style={{ flex: 2 }}>
        <MapView
          agentId={agentId}
          route={route}
          alert={alert}
        />
      </div>
    </div>
  )
}

export default App
