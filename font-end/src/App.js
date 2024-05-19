import React, { useRef, useEffect, useState } from 'react';
import axios from 'axios';
import './App.css';
import Header from './Header';

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [results, setResults] = useState([
    { name: 'Dan', probability: 0, attendance: false},
    { name: 'tai', probability: 0, attendance: false }
  ]);

  useEffect(() => {
    if (navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
          videoRef.current.srcObject = stream;
        })
        .catch((error) => {
          console.error('Error accessing webcam: ', error);
        });
    } else {
      alert('Your browser does not support accessing the webcam.');
    }
  }, []);

  const startRecognition = () => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    const processFrame = () => {
      if (!videoRef.current.paused && !videoRef.current.ended) {
        context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        const dataURL = canvas.toDataURL('image/jpeg');

        axios.post('http://localhost:8000/recog', { image: dataURL })
          .then(response => {
            const updatedResults = results.map(person => {
              const recognizedPerson = response.data.find(p => p.name === person.name);
              if (recognizedPerson && recognizedPerson.probability > 0.5 && !person.attendance) {
                return { ...person, probability: recognizedPerson.probability, attendance: true };
              } else {
                return person;
              }
            });
            setResults(updatedResults);
            requestAnimationFrame(processFrame);
          })
          .catch(error => console.error('Error:', error));
      }
    };

    processFrame();
  };

  return (
    <div className="App">
      <Header />
      <div className="content">
        <div className="video-container">
          <video ref={videoRef} width="640" height="480" autoPlay />
          <button onClick={startRecognition}>Check in</button>
          <canvas ref={canvasRef} width="640" height="480" style={{ display: 'none' }}></canvas>
        </div>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Attendance</th>
              </tr>
            </thead>
            <tbody>
              {results.map((person, index) => (
                <tr key={index}>
                  <td>{index + 1}</td>
                  <td>{person.name}</td>
                  <td>{person.attendance ? '✔️' : 'O'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default App;
