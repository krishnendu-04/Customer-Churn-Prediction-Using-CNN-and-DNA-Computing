import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";
import { getAnalytics } from "firebase/analytics";

const firebaseConfig = {
  apiKey: "AIzaSyAmsSV0YKoOyRUPPDRscRH6H9dbdg5Xels",
  authDomain: "flutter-test-4a9a7.firebaseapp.com",
  projectId: "flutter-test-4a9a7",
  storageBucket: "flutter-test-4a9a7.firebasestorage.app",
  messagingSenderId: "381270605715",
  appId: "1:381270605715:web:5a2d0fabdc666f29e97266",
  measurementId: "G-QMVX1NW82E",
};

const app = initializeApp(firebaseConfig);

export const auth = getAuth(app);
export const db   = getFirestore(app);
export const analytics = getAnalytics(app);
