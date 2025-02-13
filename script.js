async function upscaleImage() {
    const fileInput = document.getElementById("imageUpload");
    const outputImage = document.getElementById("outputImage");
    const downloadBtn = document.getElementById("downloadBtn");

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
            // 기존 이미지 캔버스 생성
            const canvas = document.createElement("canvas");
            const ctx = canvas.getContext("2d");
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);

            // ONNX 모델 로드
            try {
                console.log("start loding");
                const session = await ort.InferenceSession.create("models/esrgan.onnx");
                console.log("loding complete");

                const imageData = ctx.getImageData(0, 0, img.width, img.height);
                const float32Array = new Float32Array(imageData.data.length);
                
                for (let i = 0; i < imageData.data.length; i++) {
                    float32Array[i] = imageData.data[i] / 255.0; // 정규화
                }

                const inputTensor = new ort.Tensor("float32", float32Array, [1, 3, img.height, img.width]);
                console.log("ready");

                // 업스케일링 실행
                console.log("run...);
                const results = await session.run({ input: inputTensor });
                console.log("complete!!");

                const outputData = results.output.data;

                // output img activation
                const outputCanvas = document.createElement("canvas");
                const outputCtx = outputCanvas.getContext("2d");
                outputCanvas.width = img.width * 2;
                outputCanvas.height = img.height * 2;

                const outputImageData = outputCtx.createImageData(outputCanvas.width, outputCanvas.height);
                for (let i = 0; i < outputData.length; i++) {
                    outputImageData.data[i] = outputData[i] * 255;
                }

                outputCtx.putImageData(outputImageData, 0, 0);
                outputImage.src = outputCanvas.toDataURL();
                
                // download btn activation
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
