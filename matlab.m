%% -----------------------------
% MATLAB Real-Time YOLO Tracking & Arduino Control
%% -----------------------------

% Serial port settings
arduinoPort = "/dev/tty.usbmodem14101"; % Replace with your port
baudRate = 9600;
s = serialport(arduinoPort, baudRate);
configureTerminator(s, "LF");
flush(s);

disp('🔵 Listening for coordinates from Python YOLO...');

% Camera settings (match your webcam resolution)
imgWidth = 640;
imgHeight = 480;
centerX = imgWidth / 2;
centerY = imgHeight / 2;

% Control parameters
Kp = 0.5;    % Proportional gain for speed control
maxSpeed = 255; % Max motor speed (Arduino PWM)

% Main loop
while true
    try
        % Read a line from Python YOLO
        data = readline(s);
        coords = strsplit(strtrim(data), ',');

        if numel(coords) == 2
            % Parse coordinates
            cx = str2double(coords{1});
            cy = str2double(coords{2});

            fprintf('Object at: X=%d, Y=%d\n', cx, cy);

            % Compute error relative to center
            errorX = cx - centerX;   % positive = object to the right
            errorY = centerY - cy;   % positive = object is above center

            % Simple proportional control
            speedX = Kp * errorX;
            speedY = Kp * errorY;

            % Clamp speeds to [-maxSpeed, maxSpeed]
            speedX = max(min(speedX, maxSpeed), -maxSpeed);
            speedY = max(min(speedY, maxSpeed), -maxSpeed);

            % Format command as "X,Y\n" for Arduino
            cmd = sprintf('%d,%d\n', round(speedX), round(speedY));
            write(s, cmd, 'string');

            % Optional: read Arduino response
            % resp = readline(s);
            % fprintf('Arduino: %s\n', resp);
        end
    catch ME
        fprintf('⚠️ Error: %s\n', ME.message);
    end
end
