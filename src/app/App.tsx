import { BrowserRouter, Route, Routes } from "react-router-dom";
import { NetworkEditorPage } from "../pages/network-editor-page/NetworkEditorPage";
import { HomePage } from "../pages/home-page/HomePage";
import { DatasetManagerPage } from "../pages/dataset-manager-page/DatasetManagerPage";
import { EvolutionStudioPage } from "../pages/evolution-studio-page/EvolutionStudioPage";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/sandbox" element={<NetworkEditorPage />} />
        <Route path="/dataset-manager" element={<DatasetManagerPage />} />
        <Route path="/evolution-studio" element={<EvolutionStudioPage />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App;
