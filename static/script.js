const canvas = document.getElementById('drawing-board');
const ctx = canvas.getContext('2d');


canvas.width = 224;
canvas.height = 224;


canvas.style.border = "1px solid black";
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);
let isPainting = false;
let lineWidth = 12;


const canvasRect = canvas.getBoundingClientRect();


const draw = (e) => {
    if (!isPainting) {
        return;
    }


    const x = (e.clientX - canvasRect.left);
    const y = (e.clientY - canvasRect.top);


    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';


    ctx.lineTo(x, y);
    ctx.stroke();
}


canvas.addEventListener('mousedown', (e) => {
    isPainting = true;
    const x = (e.clientX - canvasRect.left);
    const y = (e.clientY - canvasRect.top);
    ctx.beginPath();
    ctx.moveTo(x, y);
});


canvas.addEventListener('mouseup', () => {
    isPainting = false;
    ctx.beginPath();
});


canvas.addEventListener('mousemove', draw);


canvas.addEventListener('mouseleave', () => {
    isPainting = false;
});


const clearButton = document.getElementById('clear');
clearButton.addEventListener('click', () => {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
});


document.getElementById('download').addEventListener('click', function(e) {
   
    const resizedCanvas = document.createElement('canvas');
    resizedCanvas.width = 28;
    resizedCanvas.height = 28;


    const resizedCtx = resizedCanvas.getContext('2d');


    resizedCtx.fillStyle = 'white';
    resizedCtx.fillRect(0, 0, resizedCanvas.width, resizedCanvas.height);


    resizedCtx.drawImage(canvas, 0, 0, 28, 28);


    const imageData = resizedCtx.getImageData(0, 0, 28, 28);


    const dataURL = resizedCanvas.toDataURL();


    $(document).ready(function() {
        $.ajax({
            type: "POST",
            url: "/predictLetter",
            data: {
                imageBase64: dataURL
            },
            dataType: "json",  
            success: function(response) {
                $("#output").text(response.output);
            },
            error: function() {
                console.log('Error occurred');
            }
        });
    });
});
