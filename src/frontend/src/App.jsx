import { Routes, Route } from "react-router-dom";
import Chrome from "./components/Chrome.jsx";
import Landing from "./pages/Landing.jsx";
import SingleRun from "./pages/SingleRun.jsx";
import ThresholdSweep from "./pages/ThresholdSweep.jsx";
import Matrix from "./pages/Matrix.jsx";
import About from "./pages/About.jsx";

export default function App() {
  return (
    <Chrome>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/demo" element={<SingleRun />} />
        <Route path="/sweep" element={<ThresholdSweep />} />
        <Route path="/matrix" element={<Matrix />} />
        <Route path="/about" element={<About />} />
      </Routes>
    </Chrome>
  );
}
