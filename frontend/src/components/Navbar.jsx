import { Link } from "react-router-dom";

function Navbar() {
  return (
    <nav>
      <div>
        <Link to="/">Campaign ROI Estimation</Link>
      </div>

      <div>
        <Link to="/dashboard">Dashboard</Link>
        <Link to="/predict">Predict ROI</Link>
        <Link to="/history">History</Link>
        <Link to="/model">Model</Link>
      </div>
    </nav>
  );
}

export default Navbar;
