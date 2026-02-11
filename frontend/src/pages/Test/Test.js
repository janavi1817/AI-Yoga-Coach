import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs';
import React, { useRef, useState, useEffect } from 'react'
import backend from '@tensorflow/tfjs-backend-webgl'
import Webcam from 'react-webcam'
import { count } from '../../utils/music';

import Instructions from '../../components/Instrctions/Instructions';

import './Test.css'

import DropDown from '../../components/DropDown/DropDown';
import { poseImages } from '../../utils/pose_images';
import { POINTS, keypointConnections, poseList, CLASS_NO } from '../../utils/data';
import { drawPoint, drawSegment } from '../../utils/helper'
import { NavLink } from 'react-router-dom';



let skeletonColor = 'rgb(255,255,255)'

let interval

// flag variable is used to help capture the time when AI just detect 
// the pose as correct(probability more than threshold)
let flag = false


function Test() {
  const webcamRef = useRef(null)
  const canvasRef = useRef(null)


  const [startingTime, setStartingTime] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const [poseTime, setPoseTime] = useState(0)
  const [bestPerform, setBestPerform] = useState(0)
  const [currentPose, setCurrentPose] = useState('')
  const [isStartPose, setIsStartPose] = useState(false)
  const [accuracy, setAccuracy] = useState(0)


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
    interval = setInterval(() => {
      detectPose(detector, poseClassifier, countAudio)
    }, 100)
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
        if (notDetected > 8) {
          skeletonColor = 'rgb(255,255,255)'
          return
        }
        const processedInput = landmarks_to_embedding(input)
        const classification = poseClassifier.predict(processedInput)

        classification.array().then((data) => {
          function findMaxConfidenceIndex(array) {
            let maxIndex = -1;
            let maxConfidence = 0;
            for (let i = 0; i < array.length; i++) {
              if (i === 7) continue; // Skip No_Pose
              if (array[i] > maxConfidence) {
                maxConfidence = array[i];
                maxIndex = i;
              }
            }
            return { index: maxIndex, confidence: maxConfidence };
          }

          const result = findMaxConfidenceIndex(data[0]);
          const classNo = result.index;
          const confidence = result.confidence;

          if (classNo !== -1 && confidence > 0.90) { // Threshold lowered to 0.90 for better response
            for (const poseName in CLASS_NO) {
              if (CLASS_NO[poseName] === classNo) {
                setCurrentPose(poseName)
                break;
              }
            }
          }

          const acc = confidence * 100
          setAccuracy(acc.toFixed(2))

          if (data[0][classNo] > 0.70) { // Threshold for "Correct" pose
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
    setIsStartPose(true)
    runMovenet()
  }

  function stopPose() {
    setIsStartPose(false)
    clearInterval(interval)
  }



  if (isStartPose) {
    return (
      <div className="yoga-container">
        <div className="performance-container">
          <div className="pose-performance">
            <h4>Pose Name: {currentPose || 'Scanning...'} </h4>
          </div>
          <div className="pose-performance">
            <h4>Accuracy: {accuracy} %</h4>
          </div>
        </div>

        <div className="camera-container">
          <Webcam
            id="webcam"
            ref={webcamRef}
            className="camera-view"
            style={{ zIndex: 1 }}
          />
          <canvas
            ref={canvasRef}
            id="my-canvas"
            className="camera-view"
            style={{ zIndex: 2 }}
          >
          </canvas>
        </div>

        <div>
          {currentPose && poseImages[currentPose] ? (
            <img
              src={poseImages[currentPose]}
              className="pose-img"
              alt="Pose Reference"
            />
          ) : null}
        </div>

        <div className="control-buttons">
          <button
            onClick={stopPose}
            className="secondary-btn"
          >Stop Test</button>
        </div>

      </div>
    )
  }

  return (
    <div
      className="yoga-container"
    >
      <div className="test-setup-card glass">
        <h2>Yoga Proficiency Test</h2>
        <p>Start a pose and let the AI identify it!</p>
        <div className="control-buttons">
          <button
            onClick={startYoga}
            className="secondary-btn"
          >Start Pose</button>
          <NavLink to={'/'}>
            <button className='secondary-btn'>Home</button>
          </NavLink>
        </div>
      </div>
    </div>
  )
}

export default Test