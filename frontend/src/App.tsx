import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import Header from './components/header/header';
import Form from './components/form/form'
import Text_area from './components/text-area/text-area';
import './App.css'; 


function App() {
  return (
    <div className="center-container">
      <div>
        <Header />

        {/* Main Section: Tab + Form/Text_area side by side */}
        <div className="form-section">
          {/* Left: Could be a logo or flag, or just empty */}
          <div className="logo-section">


          </div>
          {/* Right: Tab system with forms */}
          <div style={{ flex: 1 }}>
            <Tabs defaultValue="tab1" className="tabs">
              <TabsList>
                <TabsTrigger value="tab1">Form</TabsTrigger>
                <TabsTrigger value="tab2">Text Box</TabsTrigger>
              </TabsList>
              <TabsContent value="tab1">
                <Form />
              </TabsContent>
              <TabsContent value="tab2">
                <Text_area />
              </TabsContent>
            </Tabs>
          </div>
        </div>
        <div className="result-section">
        <h2>Job validation result</h2>
        <ul>
          <li>Lorem ipsum dolor sit amet, consectetur adipiscing elit.</li>
          <li>Lorem ipsum dolor sit amet, consectetur adipiscing elit.</li>
          <li>Lorem ipsum dolor sit amet, consectetur adipiscing elit.</li>
        </ul>
      </div>
      </div>
      {/* Optional: Result block below */}
      
    </div>
  );
}

export default App;