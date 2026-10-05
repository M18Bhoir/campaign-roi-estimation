import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";

import Navbar from "./components/Navbar";

import Dashboard from "./pages/Dashboard";
import Predict from "./pages/Predict";
import History from "./pages/History";
import Model from "./pages/Model";
import Prediction from "./pages/Predict";
import PredictionDetails from "./pages/PredictionDetails";

function App() {
  return (
    <BrowserRouter>
      <Navbar />

      <main>
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />

          <Route path="/dashboard" element={<Dashboard />} />

          <Route path="/predict" element={<Predict />} />

          <Route path="/history" element={<History />} />

          <Route path="/model" element={<Model />} />

          <Route path="/prediction" element={<Prediction />} />

          <Route
            path="/history/:predictionId"
            element={<PredictionDetails />}
          />
        </Routes>
      </main>
    </BrowserRouter>
  );
}

export default App;
