import ReactDOM from "react-dom/client";
import App from "./app/App";
import "./app/App.css";
import "./shared/styles/variables.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  // <React.StrictMode>
    <App />
  // </React.StrictMode>,
);
