import { BrowserRouter, Route, Routes } from "react-router-dom";
import { NetworkEditorPage } from "../pages/network-editor-page/NetworkEditorPage";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<NetworkEditorPage />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App;
