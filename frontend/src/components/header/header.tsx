import canadaflag from "@/components/header/canada.png"

function Header() {
  return (
    <header className="logo-section">
      <img src={canadaflag} alt="Canada flag" />
      <h1>Job Validation</h1>
      <hr style={{ marginTop: 16, marginBottom: 10, border: "none", borderTop: "2px solid #EEEEEE" }} />
    </header>
  );
}

export default Header;
