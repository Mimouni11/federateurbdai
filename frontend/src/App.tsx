import { useState } from "react";
import { Splash } from "./components/Splash";
import { Detector } from "./components/Detector";
import "./App.css";

export default function App() {
  const [showSplash, setShowSplash] = useState(true);
  return showSplash ? <Splash onDone={() => setShowSplash(false)} /> : <Detector />;
}
