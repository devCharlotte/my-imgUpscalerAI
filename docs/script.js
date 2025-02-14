async function upscaleImage() {
  const fileInput = document.getElementById("imageUpload");
  const outputImage = document.getElementById("outputImage");
  const downloadBtn = document.getElementById("downloadBtn");
  const scaleFactor = parseInt(document.getElementById("scaleFactor").value, 10);

  if (!fileInput.files.length) {
    alert("upload your img");
    return;
  }

  const file = fileInput.files[0];
  const reader = new FileReader();

  reader.onload = async function () {
    const img = new Image();
    img.src = reader.result;
    img.onload = async () => {
      // 원본 이미지 캔버스 생성
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      try {
        console.log("start loading model");
        // GitHub Pages에서 모델을 불러오기 위해 fetch 사용
        const modelResponse = await fetch("models/esrgan.onnx");
        const modelBuffer = await modelResponse.arrayBuffer();
        const session = await ort.InferenceSession.create(modelBuffer);
        console.log("model loading complete");

        // 이미지 전처리: RGBA → CHW RGB (정규화)
        const imageData = ctx.getImageData(0, 0, img.width, img.height);
        const numPixels = img.width * img.height;
        const inputData = new Float32Array(numPixels * 3);
        for (let i = 0; i < numPixels; i++) {
          inputData[i] = imageData.data[i * 4] / 255.0;               // R
          inputData[i + numPixels] = imageData.data[i * 4 + 1] / 255.0;   // G
          inputData[i + 2 * numPixels] = imageData.data[i * 4 + 2] / 255.0; // B
        }

        const inputTensor = new ort.Tensor("float32", inputData, [1, 3, img.height, img.width]);
        console.log("ready to run model");

        console.log("run...");
        const results = await session.run({ input: inputTensor });
        console.log("model run complete");

        const outputData = results.output.data;  // 가정: [1, 3, newHeight, newWidth] (CHW 순서)
        const newWidth = img.width * scaleFactor;
        const newHeight = img.height * scaleFactor;

        // 출력 캔버스 및 이미지 데이터 생성
        const outputCanvas = document.createElement("canvas");
        const outputCtx = outputCanvas.getContext("2d");
        outputCanvas.width = newWidth;
        outputCanvas.height = newHeight;
        const outputImageData = outputCtx.createImageData(newWidth, newHeight);
        const numOutputPixels = newWidth * newHeight;

        // 모델 출력(채널별 분리된 CHW)을 RGBA 형식으로 변환
        for (let i = 0; i < numOutputPixels; i++) {
          const r = outputData[i] * 255;
          const g = outputData[i + numOutputPixels] * 255;
          const b = outputData[i + 2 * numOutputPixels] * 255;
          outputImageData.data[i * 4] = r;
          outputImageData.data[i * 4 + 1] = g;
          outputImageData.data[i * 4 + 2] = b;
          outputImageData.data[i * 4 + 3] = 255; // alpha 채널
        }

        outputCtx.putImageData(outputImageData, 0, 0);
        outputImage.src = outputCanvas.toDataURL();

        // 다운로드 버튼 활성화
        downloadBtn.href = outputCanvas.toDataURL();
        downloadBtn.style.display = "block";
      } catch (error) {
        console.error("오류 발생:", error);
        alert("sorry, error");
      }
    };
  };

  reader.readAsDataURL(file);
}
