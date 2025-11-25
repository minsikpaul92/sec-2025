import { useState } from 'react';
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import Header from './components/header/header';
import Form from './components/form/form'
import Text_area from './components/text-area/text-area';
import ResultPanel from './components/result/result';
import type { ValidationResult } from '@/lib/validation';
import './App.css'; 

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

        <div className="form-section">
          <div className="logo-section">
            <div className="flag-icon">🇨🇦</div>
          </div>
          
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

        {result && (
          <ResultPanel result={result} statusMessage={resultMessage} />
        )}
      </div>
    </div>
  );
}

export default App;
