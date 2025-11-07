import 'dotenv/config';

const googleKey = process.env.GOOGLE_MAPS_API_KEY || "YOUR_GOOGLE_MAPS_API_KEY_HERE";
const backendUrl = process.env.BACKEND_URL || "http://127.0.0.1:5000";

console.log(googleKey === "YOUR_GOOGLE_MAPS_API_KEY_HERE" 
  ? "⚠️ Using fallback Google Maps API key (no .env found)"
  : "✅ Using Google Maps API key from .env");
console.log(backendUrl === "http://127.0.0.1:5000" 
  ? "⚠️ Using fallback BACKEND_URL (http://127.0.0.1:5000)"
  : "✅ Using BACKEND_URL from .env:", backendUrl);

export default {
  expo: {
    name: "Traffic Control UI (Web Preview)",
    slug: "traffic-control-ui-web",
    version: "1.0.0",
    sdkVersion: "54.0.0",
    platforms: ["web"],
    extra: {
      backendUrl
    },
    web: {
      favicon: "./public/vite.svg"
    }
  }
};
