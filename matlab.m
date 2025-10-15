% -----------------------------
% Setup serial port to Arduino
% -----------------------------
arduinoPort = "/dev/tty.usbmodem14101"; % Your Arduino port
baudRate = 9600;
s = serialport(arduinoPort, baudRate);
configureTerminator(s, "LF");
flush(s);

disp('Listening for coordinates from Python...');

% -----------------------------
% Main loop
% -----------------------------
while true
    % Read coordinates from Python via serial
    data = readline(s);  % Expects 'cx,cy' format
    coords = strsplit(strtrim(data), ',');

    if numel(coords) == 2
        cx = str2double(coords{1});
        cy = str2double(coords{2});
        fprintf('Received coordinates: X=%d, Y=%d\n', cx, cy);

        % -------------------------
        % Convert coordinates to commands
        % Example: simple proportional control
        % -------------------------
        imgWidth = 640;  % Match your webcam resolution
        imgHeight = 480;
        centerX = imgWidth / 2;
        centerY = imgHeight / 2;

        % Calculate error
        errorX = cx - centerX;
        errorY = cy - centerY;

        % Scale or clamp values for robot command
        speedX = int16(errorX);  % Example: send as int16
        speedY = int16(errorY);

        % -------------------------
        % Send command to Arduino
        % Format: 'X,Y\n'
        % -------------------------
        cmd = sprintf('%d,%d\n', speedX, speedY);
        write(s, cmd, 'string');

        % Optional: read Arduino response
        % resp = readline(s);
        % fprintf('Arduino: %s\n', resp);
    end
end
