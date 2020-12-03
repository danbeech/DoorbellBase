# import the necessary packages
import imagezmq
import cv2

# initialize the ImageHub object
imageHub = imagezmq.ImageHub()

# start looping over all the frames
while True:
  # receive RPi name and frame from the RPi and acknowledge
  # the receipt
  (rpiName, frame) = imageHub.recv_image()
  imageHub.send_reply(b'OK')

  # show received images
  cv2.imshow(rpiName, frame)

  # detect any kepresses
  key = cv2.waitKey(1) & 0xFF

  # if the `q` key was pressed, break from the loop
  if key == ord("q"):
    break

# do a bit of cleanup
cv2.destroyAllWindows()