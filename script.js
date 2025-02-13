async function upscaleImage() {
    const fileInput = document.getElementById("imageUpload");
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
            const canvas = document.createElement("canvas");
            const ctx = canvas.getContext("2d");
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
            
            // ONNX model load
            const session = await ort.InferenceSession.create("./models/esrgan.onnx");
            const imageData = ctx.getImageData(0, 0, img.width, img.height);
            const inputTensor = new ort.Tensor("float32", new Float32Array(imageData.data), [1, 3, img.height, img.width]);

            // run  model
            const results = await session.run({ input: inputTensor });
            const outputData = results.output.data;

            // output processing
            const outputCanvas = document.createElement("canvas");
            outputCanvas.width = img.width * 2;
            outputCanvas.height = img.height * 2;
            const outputCtx = outputCanvas.getContext("2d");
            const outputImageData = outputCtx.createImageData(outputCanvas.width, outputCanvas.height);
            outputImageData.data.set(outputData);
            outputCtx.putImageData(outputImageData, 0, 0);
            
            document.getElementById("outputImage").src = outputCanvas.toDataURL();
        };
    };

    reader.readAsDataURL(file);
}
