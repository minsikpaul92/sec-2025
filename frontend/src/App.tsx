import { Button } from './components/ui/button'

function App() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="text-center space-y-6">
        <h1 className="text-4xl font-bold text-foreground">Welcome to sec-2025</h1>
        <p className="text-xl text-muted-foreground">Built with React, Vite, and shadcn/ui</p>
        <Button size="lg">Get Started</Button>
      </div>
    </div>
  )
}

export default App
