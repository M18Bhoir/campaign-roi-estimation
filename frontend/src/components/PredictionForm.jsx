import { useState } from "react";
import { predictROI } from "../services/api";

const initialForm = {
  campaign_name: "",
  marketing_channel: "",
  target_audience: "",

  campaign_start_date: "",
  campaign_end_date: "",

  budget: "",
  competitor_score: "",
  discount_percent: "",

  platform: "",
  region: "",
  device: "",
  customer_segment: "",
  product_category: "",
  campaign_type: "",
  season: "",
  marketing_objective: "",
};

function PredictionForm({ onPrediction }) {
  const [form, setForm] = useState(initialForm);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (event) => {
    const { name, value } = event.target;

    setForm((previous) => ({
      ...previous,
      [name]: value,
    }));

    setError("");
  };

  const calculateDuration = () => {
    if (!form.campaign_start_date || !form.campaign_end_date) {
      return "";
    }

    const start = new Date(`${form.campaign_start_date}T00:00:00`);

    const end = new Date(`${form.campaign_end_date}T00:00:00`);

    const difference = Math.round((end - start) / (1000 * 60 * 60 * 24)) + 1;

    return difference > 0 ? difference : "";
  };

  const buildPayload = () => {
    const startDate = new Date(`${form.campaign_start_date}T00:00:00`);

    const durationDays = calculateDuration();

    const campaignYear = startDate.getFullYear();

    const campaignMonth = startDate.getMonth() + 1;

    const campaignQuarter = Math.ceil(campaignMonth / 3);

    /*
     * JavaScript:
     * Sunday = 0
     * Monday = 1
     * Tuesday = 2
     * ...
     *
     * Pandas:
     * Monday = 0
     * Tuesday = 1
     * ...
     * Sunday = 6
     *
     * Convert JavaScript day to Pandas day.
     */
    const campaignDayOfWeek = (startDate.getDay() + 6) % 7;

    return {
      campaign_name: form.campaign_name.trim(),

      marketing_channel: form.marketing_channel,

      target_audience: form.target_audience.trim(),

      campaign_start_date: form.campaign_start_date,

      campaign_end_date: form.campaign_end_date,

      campaign_duration_days: Number(durationDays),

      budget: Number(form.budget),

      competitor_score: Number(form.competitor_score),

      discount_percent: Number(form.discount_percent),

      platform: form.platform,

      region: form.region,

      device: form.device,

      customer_segment: form.customer_segment,

      product_category: form.product_category,

      campaign_type: form.campaign_type,

      season: form.season,

      marketing_objective: form.marketing_objective,

      campaign_year: campaignYear,

      campaign_month: campaignMonth,

      campaign_quarter: campaignQuarter,

      campaign_day_of_week: campaignDayOfWeek,
    };
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    setError("");

    if (!form.campaign_name.trim()) {
      setError("Please enter a campaign name.");
      return;
    }

    if (!form.marketing_channel) {
      setError("Please select a marketing channel.");
      return;
    }

    if (!form.target_audience.trim()) {
      setError("Please enter the target audience.");
      return;
    }

    if (!form.campaign_start_date || !form.campaign_end_date) {
      setError("Please select campaign start and end dates.");
      return;
    }

    const durationDays = calculateDuration();

    if (!durationDays) {
      setError("Please select valid campaign dates.");
      return;
    }

    if (durationDays <= 0) {
      setError("Campaign end date must be on or after the start date.");
      return;
    }

    if (form.budget === "") {
      setError("Please enter the campaign budget.");
      return;
    }

    if (Number(form.budget) < 0) {
      setError("Budget cannot be negative.");
      return;
    }

    if (form.competitor_score === "") {
      setError("Please enter the competitor score.");
      return;
    }

    if (
      Number(form.competitor_score) < 0 ||
      Number(form.competitor_score) > 100
    ) {
      setError("Competitor score must be between 0 and 100.");
      return;
    }

    if (form.discount_percent === "") {
      setError("Please enter the discount percentage.");
      return;
    }

    if (
      Number(form.discount_percent) < 0 ||
      Number(form.discount_percent) > 100
    ) {
      setError("Discount percentage must be between 0 and 100.");
      return;
    }

    try {
      setLoading(true);

      const payload = buildPayload();

      console.log("Prediction payload:", payload);

      const result = await predictROI(payload);

      console.log("Prediction response:", result);

      if (result.success === false) {
        setError(result.error?.message || "Prediction failed.");
        return;
      }

      onPrediction(result, {
        ...form,
        duration_days: durationDays,
      });
    } catch (err) {
      console.error("Prediction error:", err);

      if (err.response?.data?.error?.message) {
        setError(err.response.data.error.message);
      } else {
        setError(
          "Unable to generate prediction. Please check that the backend is running.",
        );
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <form className="prediction-form" onSubmit={handleSubmit}>
      {/* ========================= */}
      {/* CAMPAIGN INFORMATION */}
      {/* ========================= */}

      <section className="form-section">
        <div className="form-section-header">
          <p>CAMPAIGN INFORMATION</p>

          <h2>Campaign Details</h2>
        </div>

        <div className="form-grid">
          <div className="form-group form-group-full">
            <label>Campaign Name</label>

            <input
              type="text"
              name="campaign_name"
              value={form.campaign_name}
              onChange={handleChange}
              placeholder="e.g. Diwali Electronics Campaign"
              required
            />
          </div>

          <div className="form-group">
            <label>Marketing Channel</label>

            <select
              name="marketing_channel"
              value={form.marketing_channel}
              onChange={handleChange}
              required
            >
              <option value="">Select channel</option>

              <option value="Digital">Digital</option>

              <option value="Social Media">Social Media</option>

              <option value="Email">Email</option>

              <option value="Search">Search</option>

              <option value="Television">Television</option>

              <option value="Print">Print</option>

              <option value="Influencer">Influencer</option>
            </select>
          </div>

          <div className="form-group">
            <label>Target Audience</label>

            <input
              type="text"
              name="target_audience"
              value={form.target_audience}
              onChange={handleChange}
              placeholder="e.g. Young Adults"
              required
            />
          </div>
        </div>
      </section>

      {/* ========================= */}
      {/* CAMPAIGN PLANNING */}
      {/* ========================= */}

      <section className="form-section">
        <div className="form-section-header">
          <p>CAMPAIGN PLANNING</p>

          <h2>Duration & Budget</h2>
        </div>

        <div className="form-grid">
          <div className="form-group">
            <label>Start Date</label>

            <input
              type="date"
              name="campaign_start_date"
              value={form.campaign_start_date}
              onChange={handleChange}
              min="2023-01-01"
              required
            />
          </div>

          <div className="form-group">
            <label>End Date</label>

            <input
              type="date"
              name="campaign_end_date"
              value={form.campaign_end_date}
              onChange={handleChange}
              min="2023-01-01"
              required
            />
          </div>

          <div className="form-group">
            <label>Campaign Duration</label>

            <div className="readonly-input">
              {calculateDuration()
                ? `${calculateDuration()} days`
                : "Calculated automatically"}
            </div>
          </div>

          <div className="form-group">
            <label>Budget</label>

            <input
              type="number"
              name="budget"
              value={form.budget}
              onChange={handleChange}
              min="0"
              step="0.01"
              placeholder="Enter budget"
              required
            />
          </div>

          <div className="form-group">
            <label>Competitor Score</label>

            <input
              type="number"
              name="competitor_score"
              value={form.competitor_score}
              onChange={handleChange}
              min="0"
              max="100"
              step="0.01"
              placeholder="0 - 100"
              required
            />
          </div>

          <div className="form-group">
            <label>Discount %</label>

            <input
              type="number"
              name="discount_percent"
              value={form.discount_percent}
              onChange={handleChange}
              min="0"
              max="100"
              step="0.01"
              placeholder="0 - 100"
              required
            />
          </div>
        </div>
      </section>

      {/* ========================= */}
      {/* TARGETING & STRATEGY */}
      {/* ========================= */}

      <section className="form-section">
        <div className="form-section-header">
          <p>CAMPAIGN CHARACTERISTICS</p>

          <h2>Targeting & Strategy</h2>
        </div>

        <div className="form-grid">
          {/* PLATFORM */}

          <div className="form-group">
            <label>Platform</label>

            <select
              name="platform"
              value={form.platform}
              onChange={handleChange}
              required
            >
              <option value="">Select platform</option>

              <option value="Facebook">Facebook</option>

              <option value="Google Ads">Google Ads</option>

              <option value="Instagram">Instagram</option>

              <option value="LinkedIn">LinkedIn</option>

              <option value="TikTok">TikTok</option>

              <option value="Twitter (X)">Twitter (X)</option>

              <option value="YouTube">YouTube</option>
            </select>
          </div>

          {/* REGION */}

          <div className="form-group">
            <label>Region</label>

            <select
              name="region"
              value={form.region}
              onChange={handleChange}
              required
            >
              <option value="">Select region</option>

              <option value="Africa">Africa</option>

              <option value="Asia Pacific">Asia Pacific</option>

              <option value="Europe">Europe</option>

              <option value="Middle East">Middle East</option>

              <option value="North America">North America</option>

              <option value="South America">South America</option>
            </select>
          </div>

          {/* DEVICE */}

          <div className="form-group">
            <label>Device</label>

            <select
              name="device"
              value={form.device}
              onChange={handleChange}
              required
            >
              <option value="">Select device</option>

              <option value="Desktop">Desktop</option>

              <option value="Mobile">Mobile</option>

              <option value="Smart TV">Smart TV</option>

              <option value="Tablet">Tablet</option>
            </select>
          </div>

          {/* CUSTOMER SEGMENT */}

          <div className="form-group">
            <label>Customer Segment</label>

            <select
              name="customer_segment"
              value={form.customer_segment}
              onChange={handleChange}
              required
            >
              <option value="">Select segment</option>

              <option value="Enterprise">Enterprise</option>

              <option value="New">New</option>

              <option value="Premium">Premium</option>

              <option value="Returning">Returning</option>

              <option value="Senior">Senior</option>

              <option value="Student">Student</option>
            </select>
          </div>

          {/* PRODUCT CATEGORY */}

          <div className="form-group">
            <label>Product Category</label>

            <select
              name="product_category"
              value={form.product_category}
              onChange={handleChange}
              required
            >
              <option value="">Select category</option>

              <option value="Automobile">Automobile</option>

              <option value="Beauty">Beauty</option>

              <option value="Education">Education</option>

              <option value="Electronics">Electronics</option>

              <option value="Fashion">Fashion</option>

              <option value="Finance">Finance</option>

              <option value="Food">Food</option>

              <option value="Healthcare">Healthcare</option>

              <option value="Real Estate">Real Estate</option>

              <option value="Travel">Travel</option>
            </select>
          </div>

          {/* CAMPAIGN TYPE */}

          <div className="form-group">
            <label>Campaign Type</label>

            <select
              name="campaign_type"
              value={form.campaign_type}
              onChange={handleChange}
              required
            >
              <option value="">Select type</option>

              <option value="App Promotion">App Promotion</option>

              <option value="Awareness">Awareness</option>

              <option value="Lead Generation">Lead Generation</option>

              <option value="Product Launch">Product Launch</option>

              <option value="Retention">Retention</option>

              <option value="Sales">Sales</option>

              <option value="Seasonal Promotion">Seasonal Promotion</option>
            </select>
          </div>

          {/* SEASON */}

          <div className="form-group">
            <label>Season</label>

            <select
              name="season"
              value={form.season}
              onChange={handleChange}
              required
            >
              <option value="">Select season</option>

              <option value="Autumn">Autumn</option>

              <option value="Festival">Festival</option>

              <option value="Holiday">Holiday</option>

              <option value="Spring">Spring</option>

              <option value="Summer">Summer</option>

              <option value="Winter">Winter</option>
            </select>
          </div>

          {/* MARKETING OBJECTIVE */}

          <div className="form-group">
            <label>Marketing Objective</label>

            <select
              name="marketing_objective"
              value={form.marketing_objective}
              onChange={handleChange}
              required
            >
              <option value="">Select objective</option>

              <option value="Brand Awareness">Brand Awareness</option>

              <option value="Lead Generation">Lead Generation</option>

              <option value="Retention">Retention</option>

              <option value="Sales Growth">Sales Growth</option>
            </select>
          </div>
        </div>
      </section>

      {/* ========================= */}
      {/* ERROR */}
      {/* ========================= */}

      {error && <div className="prediction-error">{error}</div>}

      {/* ========================= */}
      {/* SUBMIT */}
      {/* ========================= */}

      <div className="prediction-submit">
        <button type="submit" disabled={loading}>
          {loading ? "Generating Prediction..." : "Generate ROI Prediction"}
        </button>
      </div>
    </form>
  );
}

export default PredictionForm;
