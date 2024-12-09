// Penguin objects
const penguins = [
    { x: 100, y: 200 },
    { x: 300, y: 150 },
    { x: 400, y: 500 },
    { x: 600, y: 750 },
    { x: 900, y: 300 },
    { x: 1100, y: 250 },
    { x: 1300, y: 456 },
    { x: 1600, y: 975 },
];

// Create penguin elements and add them to the DOM
penguins.forEach(penguin => {
    const img = document.createElement('img');
    img.src = 'penguin.png';
    img.classList.add('penguin');
    img.style.left = penguin.x + 'px';
    img.style.top = penguin.y + 'px';
    document.body.appendChild(img);
});

// Handle click events
document.addEventListener('click', (event) => {
	throwSnowball(event.clientX, event.clientY);
});

function goFullscreen() {
  const element = document.documentElement;
  if (element.requestFullscreen) {
    element.requestFullscreen();
  } else if (element.mozRequestFullScreen)  
 {
    element.mozRequestFullScreen();
  } else if (element.webkitRequestFullscreen) {
    element.webkitRequestFullscreen();
  } else if (element.msRequestFullscreen) {
    element.msRequestFullscreen();  

  }
}

document.addEventListener('keydown', (event) => {
  if (event.key === 'f') {
    goFullscreen();
  }
});


let snowballs = [];

function throwSnowball(x, y) {

    penguins.forEach(penguin => {
        if (x >= penguin.x && x <= penguin.x + 100 &&
            y >= penguin.y && y <= penguin.y + 100) {
            const img = document.querySelector(`img[style="left: ${penguin.x}px; top: ${penguin.y}px;"]`);
            img.src = 'penguin_hit.png';
            setTimeout(() => {
                img.remove();
            }, 1000);
        }
    });


    const snowball = document.createElement('img');
    snowball.src = 'snowball.png';
    snowball.classList.add('snowball');
    snowball.style.left = (-117/2) + x + 'px';
    snowball.style.top = (-134/2) + y + 'px';
    document.body.appendChild(snowball);

    snowballs.push(snowball);

    let yPosition = (-134/2) + y;
    const intervalId = setInterval(() => {
        yPosition += 5;
        snowball.style.top = yPosition + 'px';

        if (yPosition > window.innerHeight) {
            clearInterval(intervalId);
            snowball.remove();
            snowballs = snowballs.filter(ball => ball !== snowball);
        }
    }, 10);
}

var socket = io('localhost:5000', {transports: ['websocket']});
socket.on('message', function(message) {
    console.log(message);
    // message has x_px, y_px, xr, yr
    // xr and yr are the ratio of the screen dimensions to hit
    let x = Math.round(window.screen.width * message.xr);
    let y = Math.round(window.screen.height * message.yr);
    throwSnowball(x, y);
});

socket.on('heartbeat', function(message) {
    console.log(message);
});


//window.onload = function() {
//    const eventSource = new EventSource('/api/events');
//
//    eventSource.onmessage = (event) => {
//        console.log('Received data:', event.data);
//        throwSnowball(200, 200);
//    };
//
//    eventSource.onerror = (error) => {
//      console.error('Error occurred:', error);
//    };
//}
