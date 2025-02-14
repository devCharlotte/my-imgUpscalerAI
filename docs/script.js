async function upscaleImage() {
  const fileInput = document.getElementById("imageUpload");
  const outputImage = document.getElementById("outputImage");
  const downloadBtn = document.getElementById("downloadBtn");
  const scaleFactor = parseInt(document.getElementById("scaleFactor").value, 10);

  if (!fileInput.files.length) {
    alert("Please upload your image.");
    return;
  }

  const file = fileInput.files[0];
  const reader = new FileReader();

  reader.onload = async function() {
    const img = new Image();
    img.src = reader.result;
    img.onload = async () => {
      // 원본 이미지를 그릴 캔버스 생성
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      try {
        console.log("Start loading Real-ESRGAN model");

        // 외부에 호스팅된 ONNX 모델 URL (예시 URL, 실제 변환 후 업로드한 URL로 교체)
        const modelURL = "https://raw.githubusercontent.com/your-username/your-model-repo/main/RealESRGAN_x4plus.onnx";
        const modelResponse = await fetch(modelURL);
        if (!modelResponse.ok) {
          throw new Error(`Model fetch failed: ${modelResponse.statusText}`);
        }
        const modelBuffer = await modelResponse.arrayBuffer();
        const session = await ort.InferenceSession.create(modelBuffer);
        console.log("Model loaded");
        console.log("Input names:", session.inputNames);
        console.log("Output names:", session.outputNames);

        // 이미지 전처리: RGBA → CHW (RGB만 사용, 0~1로 정규화)
        const imageData = ctx.getImageData(0, 0, img.width, img.height);
        const numPixels = img.width * img.height;
        const inputData = new Float32Array(numPixels * 3);
        for (let i = 0; i < numPixels; i++) {
          inputData[i] = imageData.data[i * 4] / 255.0;                      // R 채널
          inputData[i + numPixels] = imageData.data[i * 4 + 1] / 255.0;          // G 채널
          inputData[i + 2 * numPixels] = imageData.data[i * 4 + 2] / 255.0;        // B 채널
        }

        // 모델의 첫 번째 입력 이름 사용 (없으면 "input" 기본값 사용)
        const inputName = session.inputNames[0] || "input";
        const inputTensor = new ort.Tensor("float32", inputData, [1, 3, img.height, img.width]);
        console.log("Ready to run model");

        console.log("Running model...");
        const results = await session.run({ [inputName]: inputTensor });
        console.log("Model run complete");

        // 모델의 첫 번째 출력 이름 사용 (없으면 "output" 기본값)
        const outputName = session.outputNames[0] || "output";
        const outputTensor = results[outputName];
        const outputData = outputTensor.data; // 가정: [1, 3, newHeight, newWidth]

        // 선택한 스케일로 출력 크기 결정 (모델이 x4 등 고정 배율인 경우 별도 후처리 필요)
        const newWidth = img.width * scaleFactor;
        const newHeight = img.height * scaleFactor;

        // 후처리: CHW → RGBA 변환
        const outputCanvas = document.createElement("canvas");
        const outputCtx = outputCanvas.getContext("2d");
        outputCanvas.width = newWidth;
        outputCanvas.height = newHeight;
        const outputImageData = outputCtx.createImageData(newWidth, newHeight);
        const numOutputPixels = newWidth * newHeight;

        for (let i = 0; i < numOutputPixels; i++) {
          const r = outputData[i] * 255;
          const g = outputData[i + numOutputPixels] * 255;
          const b = outputData[i + 2 * numOutputPixels] * 255;
          outputImageData.data[i * 4] = r;
          outputImageData.data[i * 4 + 1] = g;
          outputImageData.data[i * 4 + 2] = b;
          outputImageData.data[i * 4 + 3] = 255; // Alpha 채널
        }

        outputCtx.putImageData(outputImageData, 0, 0);
        outputImage.src = outputCanvas.toDataURL();
        downloadBtn.href = outputCanvas.toDataURL();
        downloadBtn.style.display = "block";
      } catch (error) {
        console.error("Error occurred:", error);
        alert("Error occurred: " + error.message);
      }
    };
  };

  reader.readAsDataURL(file);
}
