import { useState } from "react";

import PredictionForm from "../components/PredictionForm";
import PredictionResult from "../components/PredictionResult";

function Predict() {
  const [prediction, setPrediction] = useState(null);
  const [campaignData, setCampaignData] = useState(null);

  const handlePrediction = (result, campaign) => {
    setPrediction(result);
    setCampaignData(campaign);
  };

  const handleReset = () => {
    setPrediction(null);
    setCampaignData(null);
  };

  return (
    <div className="prediction-page">
      <section className="prediction-header">
        <div>
          <p className="eyebrow">CAMPAIGN PREDICTION</p>

          <h1>Predict Campaign ROI</h1>

          <p>
            Enter your planned campaign information to estimate its expected
            return on investment.
          </p>
        </div>
      </section>

      {!prediction ? (
        <PredictionForm onPrediction={handlePrediction} />
      ) : (
        <PredictionResult
          result={prediction}
          campaign={campaignData}
          onReset={handleReset}
        />
      )}
    </div>
  );
}

export default Predict;
