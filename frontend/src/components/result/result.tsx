import type { ValidationResult } from "@/lib/validation";

interface ResultProps {
  result: ValidationResult;
  statusMessage: string;
}

const ResultPanel = ({ result, statusMessage }: ResultProps) => {
  const isValid = result.classification?.toUpperCase() === "VALID";
  const missingFields = result.missing_fields ?? [];
  const transparencyScore =
    typeof result.transparency_score === "number"
      ? result.transparency_score
      : null;
  const hasMissing = missingFields.length > 0;

  const formatPercent = (value?: number) =>
    typeof value === "number" ? `${value.toFixed(1)}%` : "--";

  const prettifyField = (field: string) =>
    field
      .split("_")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");

  return (
    <section className="result-section">
      <div className={`result-banner ${isValid ? "valid" : "invalid"}`}>
        <div className="banner-icon" aria-hidden="true">
          {isValid ? "✅" : "⚠️"}
        </div>
        <div>
          <p className="banner-label">
            {isValid ? "All clear" : "Action required"}
          </p>
          <p className="banner-message">{statusMessage}</p>
        </div>
      </div>

      <div className="result-metrics">
        <div className="metric-card">
          <p className="metric-label">Confidence</p>
          <p className="metric-value">
            {formatPercent(result.confidence ?? 0)}
          </p>
          <p className="metric-subtext">Model certainty</p>
        </div>
        <div className="metric-card">
          <p className="metric-label">Transparency score</p>
          <p className="metric-value">
            {transparencyScore !== null
              ? formatPercent(transparencyScore)
              : "--"}
          </p>
          <p className="metric-subtext">Disclosure completeness</p>
        </div>
        <div className="metric-card">
          <p className="metric-label">Missing fields</p>
          <p
            className={`metric-value ${
              hasMissing ? "metric-alert" : "metric-success"
            }`}
          >
            {hasMissing ? missingFields.length : "0"}
          </p>
          <p className="metric-subtext">
            {hasMissing
              ? "Provide details listed below"
              : "All mandatory fields present"}
          </p>
        </div>
      </div>

      {hasMissing ? (
        <div className="result-details">
          <h3>What still needs attention</h3>
          <p className="result-hint">
            Supply information for every item to reach compliance.
          </p>
          <div className="missing-pill-grid">
            {missingFields.map((field) => (
              <span className="missing-pill" key={field}>
                {prettifyField(field)}
              </span>
            ))}
          </div>
        </div>
      ) : (
        <div className="result-details success">
          <h3>Great! All mandatory fields are filled.</h3>
          <p className="result-hint">
            Submit the posting confidently—every required detail is present.
          </p>
        </div>
      )}
    </section>
  );
};

export default ResultPanel;
