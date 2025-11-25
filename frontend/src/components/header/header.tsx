import canadaflag from "@/components/header/canada.png"

function Header() {
  return (
    <header className="logo-section">
      <img src={canadaflag} alt="Canada flag" />
      {/* Main title */}
      <h1>Job Validation</h1>
      {/* Optional: Horizontal line for separation */}
      <hr style={{ marginTop: 16, marginBottom: 10, border: "none", borderTop: "2px solid #EEEEEE" }} />
    </header>
  );
}

export default Header;
