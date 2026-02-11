import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs';
import React, { useRef, useState, useEffect } from 'react'
import backend from '@tensorflow/tfjs-backend-webgl'
import Webcam from 'react-webcam'
import { count } from '../../utils/music';

import Instructions from '../../components/Instrctions/Instructions';

import './Yoga.css'

import DropDown from '../../components/DropDown/DropDown';
import { poseImages } from '../../utils/pose_images';
import { POINTS, keypointConnections, poseList, CLASS_NO } from '../../utils/data';
import { drawPoint, drawSegment } from '../../utils/helper'
import { Route, NavLink } from 'react-router-dom';

let skeletonColor = 'rgb(255,255,255)'

let interval

// flag variable is used to help capture the time when AI just detect 
// the pose as correct(probability more than threshold)
let flag = false


function Yoga() {
  const webcamRef = useRef(null)
  const canvasRef = useRef(null)


  const [startingTime, setStartingTime] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const [poseTime, setPoseTime] = useState(0)
  const [bestPerform, setBestPerform] = useState(0)
  const [currentPose, setCurrentPose] = useState('Vrukshasana')
  const [isStartPose, setIsStartPose] = useState(false)
  const [accuracy, setAccuracy] = useState(0)
  const [isImageMode, setIsImageMode] = useState(false)
  const [capturedImage, setCapturedImage] = useState(null)
  const imgRef = useRef(null)
  const fileInputRef = useRef(null)


  useEffect(() => {
    const timeDiff = (currentTime - startingTime) / 1000
    if (flag) {
      setPoseTime(timeDiff)
    }
    if ((currentTime - startingTime) / 1000 > bestPerform) {
      setBestPerform(timeDiff)
    }
  }, [currentTime])


  useEffect(() => {
    setCurrentTime(0)
    setPoseTime(0)
    setBestPerform(0)
  }, [currentPose])


  function get_center_point(landmarks, left_bodypart, right_bodypart) {
    let left = tf.gather(landmarks, left_bodypart, 1)
    let right = tf.gather(landmarks, right_bodypart, 1)
    const center = tf.add(tf.mul(left, 0.5), tf.mul(right, 0.5))
    return center

  }

  function get_pose_size(landmarks, torso_size_multiplier = 2.5) {
    let hips_center = get_center_point(landmarks, POINTS.LEFT_HIP, POINTS.RIGHT_HIP)
    let shoulders_center = get_center_point(landmarks, POINTS.LEFT_SHOULDER, POINTS.RIGHT_SHOULDER)
    let torso_size = tf.norm(tf.sub(shoulders_center, hips_center))
    let pose_center_new = get_center_point(landmarks, POINTS.LEFT_HIP, POINTS.RIGHT_HIP)
    pose_center_new = tf.expandDims(pose_center_new, 1)

    pose_center_new = tf.broadcastTo(pose_center_new,
      [1, 17, 2]
    )
    // return: shape(17,2)
    let d = tf.gather(tf.sub(landmarks, pose_center_new), 0, 0)
    let max_dist = tf.max(tf.norm(d, 'euclidean', 1))

    // normalize scale
    let pose_size = tf.maximum(tf.mul(torso_size, torso_size_multiplier), max_dist)
    return pose_size
  }

  function normalize_pose_landmarks(landmarks) {
    let pose_center = get_center_point(landmarks, POINTS.LEFT_HIP, POINTS.RIGHT_HIP)
    pose_center = tf.expandDims(pose_center, 1)
    pose_center = tf.broadcastTo(pose_center,
      [1, 17, 2]
    )
    landmarks = tf.sub(landmarks, pose_center)

    let pose_size = get_pose_size(landmarks)
    landmarks = tf.div(landmarks, pose_size)
    return landmarks
  }

  function landmarks_to_embedding(landmarks) {
    // normalize landmarks 2D
    landmarks = normalize_pose_landmarks(tf.expandDims(landmarks, 0))
    let embedding = tf.reshape(landmarks, [1, 34])
    return embedding
  }
  //https://models.s3.jp-tok.cloud-object-storage.appdomain.cloud/model.json
  const runMovenet = async () => {
    const detectorConfig = { modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER };
    const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
    const poseClassifier = await tf.loadLayersModel('/model/model.json')
    const countAudio = new Audio(count)
    countAudio.loop = true

    // Clear any existing interval to prevent duplicates
    if (interval) clearInterval(interval);

    if (isImageMode) {
      // Small delay to ensure image is loaded in DOM
      detectionTimeout = setTimeout(() => {
        detectPoseImage(detector, poseClassifier)
      }, 1000)
    } else {
      interval = setInterval(() => {
        detectPose(detector, poseClassifier, countAudio)
      }, 100)
    }
  }

  let detectionTimeout;

  // Cleanup on unmount or mode change
  useEffect(() => {
    return () => {
      if (interval) clearInterval(interval);
      if (detectionTimeout) clearTimeout(detectionTimeout);
    }
  }, []);

  const detectPoseImage = async (detector, poseClassifier) => {
    if (
      typeof imgRef.current !== "undefined" &&
      imgRef.current !== null
    ) {
      try {
        const image = imgRef.current;
        const pose = await detector.estimatePoses(image);
        const ctx = canvasRef.current.getContext('2d');
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

        // Draw result
        const keypoints = pose[0].keypoints;
        let input = keypoints.map((keypoint) => {
          if (keypoint.score > 0.2) {
            if (!(keypoint.name === 'left_eye' || keypoint.name === 'right_eye')) {
              drawPoint(ctx, keypoint.x, keypoint.y, 8, 'rgb(255,255,255)');
              let connections = keypointConnections[keypoint.name];
              try {
                connections.forEach((connection) => {
                  let conName = connection.toUpperCase();
                  drawSegment(ctx, [keypoint.x, keypoint.y],
                    [keypoints[POINTS[conName]].x,
                    keypoints[POINTS[conName]].y],
                    skeletonColor);
                });
              } catch (err) {
              }
            }
          }
          return [keypoint.x, keypoint.y];
        });

        const processedInput = landmarks_to_embedding(input);
        const classification = poseClassifier.predict(processedInput);

        classification.array().then((data) => {
          const classNo = CLASS_NO[currentPose];

          if (classNo >= data[0].length) {
            console.warn("Model does not support this class yet");
            setAccuracy("0.00");
            return;
          }

          const acc = data[0][classNo] * 100;
          setAccuracy(acc.toFixed(2));

          if (data[0][classNo] > 0.70) {
            skeletonColor = 'rgb(0,255,0)';
          } else {
            skeletonColor = 'rgb(255,255,255)';
          }
        });

      } catch (error) {
        console.log(error);
      }
    }
  }

  const detectPose = async (detector, poseClassifier, countAudio) => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      let notDetected = 0
      const video = webcamRef.current.video
      const videoWidth = webcamRef.current.video.videoWidth
      const videoHeight = webcamRef.current.video.videoHeight
      webcamRef.current.video.width = videoWidth
      webcamRef.current.video.height = videoHeight

      canvasRef.current.width = videoWidth
      canvasRef.current.height = videoHeight

      const pose = await detector.estimatePoses(video)
      const ctx = canvasRef.current.getContext('2d')
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      try {
        const keypoints = pose[0].keypoints
        let input = keypoints.map((keypoint) => {
          if (keypoint.score > 0.2) {
            if (!(keypoint.name === 'left_eye' || keypoint.name === 'right_eye')) {
              drawPoint(ctx, keypoint.x, keypoint.y, 8, 'rgb(255,255,255)')
              let connections = keypointConnections[keypoint.name]
              try {
                connections.forEach((connection) => {
                  let conName = connection.toUpperCase()
                  drawSegment(ctx, [keypoint.x, keypoint.y],
                    [keypoints[POINTS[conName]].x,
                    keypoints[POINTS[conName]].y]
                    , skeletonColor)
                })
              } catch (err) {

              }

            }
          } else {
            notDetected += 1
          }
          return [keypoint.x, keypoint.y]
        })
        if (notDetected > 12) {
          skeletonColor = 'rgb(255,255,255)'
          return
        }
        const processedInput = landmarks_to_embedding(input)
        const classification = poseClassifier.predict(processedInput)

        classification.array().then((data) => {
          const classNo = CLASS_NO[currentPose]

          // Debugging: Show all confidences
          const predictions = Array.from(data[0]).map((conf, i) => ({
            index: i,
            confidence: conf.toFixed(4),
            name: Object.keys(CLASS_NO).find(key => CLASS_NO[key] === i) || `Index_${i}`
          })).sort((a, b) => b.confidence - a.confidence);

          console.log(`--- AI Predictions (Current: ${currentPose}) ---`);
          console.table(predictions.slice(0, 3));

          if (classNo >= data[0].length) {
            console.warn("Model does not support this class yet");
            setAccuracy("0.00");
            skeletonColor = 'rgb(255,255,255)'; // Default color
            return;
          }

          const acc = data[0][classNo] * 100
          const Acc = acc.toFixed(2)
          setAccuracy(Acc)
          if (data[0][classNo] > 0.70) {

            if (!flag) {
              countAudio.play()
              setStartingTime(new Date(Date()).getTime())
              flag = true
            }
            setCurrentTime(new Date(Date()).getTime())
            skeletonColor = 'rgb(0,255,0)'
          } else {
            flag = false
            skeletonColor = 'rgb(255,255,255)'
            countAudio.pause()
            countAudio.currentTime = 0
          }
        })
      } catch (err) {
        console.log(err)
      }


    }
  }

  function startYoga() {
    setIsImageMode(false)
    setIsStartPose(true)
    runMovenet()
  }

  const handleImageUpload = (event) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      const reader = new FileReader();
      reader.onloadend = () => {
        setCapturedImage(reader.result);
        setIsImageMode(true);
        setIsStartPose(true);
      };
      reader.readAsDataURL(file);
    }
  };

  useEffect(() => {
    if (isStartPose) {
      runMovenet();
    } else {
      if (interval) clearInterval(interval);
    }
  }, [isStartPose, isImageMode]); // Trigger when mode changes or start state changes 

  function stopPose() {
    setIsStartPose(false)
    clearInterval(interval)
  }

  if (isStartPose) {
    return (
      <div className="yoga-container">
        <div className="performance-container">
          <div className="pose-performance">
            <h4>Pose Time: {poseTime} s</h4>
          </div>
          <div className="pose-performance">
            <h4>Best: {bestPerform} s</h4>
          </div>
          <div className="pose-performance">
            <h4>Accuracy {accuracy} </h4>
          </div>
        </div>

        <div className="camera-container">
          {isImageMode ?
            <img
              ref={imgRef}
              src={capturedImage}
              alt="uploaded"
              className="camera-view"
              style={{ zIndex: 1 }} // Ensure image is behind canvas but visible
            />
            :
            <Webcam
              id="webcam"
              ref={webcamRef}
              className="camera-view"
              style={{ zIndex: 1 }}
            />
          }
          <canvas
            ref={canvasRef}
            id="my-canvas"
            className="camera-view"
            style={{ zIndex: 2 }}
          >
          </canvas>
        </div>

        <div>
          <img
            src={poseImages[currentPose]}
            className="pose-img"
            alt="Pose Reference"
          />
        </div>

        <div className="control-buttons">
          <button
            onClick={stopPose}
            className="secondary-btn"
          >Stop Pose</button>
        </div>
      </div >
    )
  }

  return (
    <div
      className="yoga-container"
    >
      <DropDown
        poseList={poseList}
        currentPose={currentPose}
        setCurrentPose={setCurrentPose}
      />
      <Instructions
        currentPose={currentPose}
      />

      <div className="control-buttons">
        <button
          onClick={startYoga}
          className="secondary-btn"
        >Start Pose (Webcam)</button>

        <input
          type="file"
          accept="image/*"
          style={{ display: 'none' }}
          ref={fileInputRef}
          onChange={handleImageUpload}
        />
        <button
          onClick={() => fileInputRef.current.click()}
          className="secondary-btn"
        >Upload Image</button>

        <NavLink to={'/'}>
          <button className='secondary-btn'>Home</button>
        </NavLink>
      </div>
    </div>
  )
}

export default Yoga