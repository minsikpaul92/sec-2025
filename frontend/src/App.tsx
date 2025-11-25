import { useState } from 'react';
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import Header from './components/header/header';
import Form from './components/form/form'
import Text_area from './components/text-area/text-area';
import './App.css'; 

interface ValidationResult {
  classification: string;
  confidence: number;
  missing_fields?: string[];
  transparency_score?: number;
}

function App() {
  const [result, setResult] = useState<ValidationResult | null>(null);
  const [resultMessage, setResultMessage] = useState<string>("");

  const handleFormResult = (data: ValidationResult) => {
    setResult(data);
    if (data.classification === "VALID") {
      setResultMessage("✓ Job posting is VALID and meets all requirements!");
    } else {
      setResultMessage("✗ Job posting is INVALID. Check the details below.");
    }
  };

  return (
    <div className="center-container">
      <div className="app-wrapper">
        <Header />

        {/* Main Section: Tab + Form/Text_area side by side */}
        <div className="form-section">
          {/* Left: Could be a logo or flag, or just empty */}
          <div className="logo-section">
            <div className="flag-icon">🇨🇦</div>
          </div>
          
          {/* Right: Tab system with forms */}
          <div className="tabs-wrapper">
            <Tabs defaultValue="tab1" className="tabs">
              <TabsList className="tabs-list">
                <TabsTrigger value="tab1">Form</TabsTrigger>
                <TabsTrigger value="tab2">Text Box</TabsTrigger>
              </TabsList>
              <TabsContent value="tab1">
                <Form onResult={handleFormResult} />
              </TabsContent>
              <TabsContent value="tab2">
                <Text_area onResult={handleFormResult} />
              </TabsContent>
            </Tabs>
          </div>
        </div>

        {/* Result Section */}
        {result && (
          <div className="result-section">
            <h2>Job validation result</h2>
            <div className={`result-status ${result.classification.toLowerCase()}`}>
              <strong>{resultMessage}</strong>
            </div>
            
            {result.missing_fields && result.missing_fields.length > 0 && (
              <div className="result-details">
                <h3>Missing Fields:</h3>
                <ul>
                  {result.missing_fields.map((field, idx) => (
                    <li key={idx}>{field}</li>
                  ))}
                </ul>
              </div>
            )}

            {result.transparency_score !== undefined && (
              <div className="result-details">
                <h3>Transparency Score: {result.transparency_score}%</h3>
              </div>
            )}

            <div className="result-confidence">
              <p>Confidence: {(result.confidence || 0).toFixed(2)}%</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;