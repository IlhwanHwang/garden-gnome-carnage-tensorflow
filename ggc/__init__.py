import pywinauto
import time
import mss
import os
import scipy.misc
import numpy as np
import win32api
import win32con
import cv2

class GGC:

	def __init__(self):

		# Start GGC
		self.app = pywinauto.Application().start(os.path.join(os.path.dirname(__file__), 'game/ggc.exe'))

		print('Loading...')

		while not self.app_load():
			time.sleep(0.0)
		
		print('GGC is on')

		self.control = self.window.wrapper_object()
		self.status = 'IDLE'

		# Hand-written GGC UI parameters
		self.width = 640
		self.height = 480
		self.score_pos = (16, 14)
		self.score_size = (12, 22)
		self.score_digits = 7
		self.score_sep = 4
		self.airstike_rect = (142, 13, 142 + 38, 13 + 22)

		# Number recognition
		im = scipy.misc.imread(os.path.join(os.path.dirname(__file__), 'number.png'), 'GRAY')
		im = np.reshape(im, [10, -1])
		im = im.astype(np.float32)
		im = im / 128.0 - 1.0
		W = (im.T / np.sum(np.abs(im), axis=1)).T
		self.number_w = W / 128.0
		self.number_b = np.reshape(-np.sum(W, axis=1), [-1, 1])

		# 'Done' recognition
		self.airstrike_orig = im = scipy.misc.imread(os.path.join(os.path.dirname(__file__), 'airstrike.png'))

		# Prepare capturing
		rect = self.control.ClientAreaRect()
		self.monitor = { 
			'left': rect.left,
			'top' : rect.top,
			'width': rect.width(),
			'height': rect.height()
			}
		self.captured = np.array([])
		self.sct = mss.mss()

		self.action_before = (False, False, False, False)
		self.done = False


	def app_load(self):
		self.window = self.app.window(title='Garden Gnome Carnage')
		if self.window.exists():
			return True
		else:
			return False


	# Transit IDLE -> GAME
	def start(self):
		if self.status == 'IDLE':
			self.control.set_focus()
			self.control.send_keystrokes('{VK_SPACE}')
			time.sleep(0.7)
			self.control.send_keystrokes('{VK_SPACE}')
			time.sleep(0.7)
			self.status = 'GAME'


	# Transit GAME -> PAUSE
	def resume(self):
		if self.status == 'PAUSE':
			# Release keys
			win32api.keybd_event(0x25, 0, win32con.KEYEVENTF_KEYUP, 0)
			win32api.keybd_event(0x27, 0, win32con.KEYEVENTF_KEYUP, 0)
			self.control.send_keystrokes('{VK_ESCAPE}')
			self.status = 'GAME'


	# Transit PAUSE -> GAME
	def pause(self):
		if self.status == 'GAME':
			self.control.send_keystrokes('{VK_ESCAPE}')
			self.status = 'PAUSE'


	# Transit DONE -> GAME
	def restart(self):
		if self.status == 'DONE':
			time.sleep(10.0)
			for _ in range(4):
				self.control.send_keystrokes('{VK_SPACE}')
				time.sleep(0.1)
			time.sleep(3.0)
			self.status = 'IDLE'
			self.start()


	def action(self, act):
		# pywinauto does not provide function to hold keyboard. So we use win32api instead.
		if np.all(self.action_before == act):
			pass
		else:
			# At first, release the previous keyboard action.
			if self.action_before[0]:
				win32api.keybd_event(0x25, 0, win32con.KEYEVENTF_KEYUP, 0)
			if self.action_before[1]:
				win32api.keybd_event(0x27, 0, win32con.KEYEVENTF_KEYUP, 0)
			if self.action_before[2]:
				win32api.keybd_event(0x20, 0, win32con.KEYEVENTF_KEYUP, 0)

			# Secondly, hold newly queued keyboard actions.
			if act[0]:
				win32api.keybd_event(0x25, 0, 0, 0)
			if act[1]:
				win32api.keybd_event(0x27, 0, 0, 0)
			if act[2]:
				win32api.keybd_event(0x20, 0, 0, 0)

		self.action_before = act


	def capture(self):
		sct_img = self.sct.grab(self.monitor)
		self.captured = np.array(sct_img)
		if self.captured.shape[2] == 4:
			self.captured = self.captured[:,:,:3]

		# Crop score region
		self.score = 0
		for i in range(self.score_digits):
			x1 = self.score_pos[0] + (self.score_sep + self.score_size[0]) * i
			y1 = self.score_pos[1]
			x2 = x1 + self.score_size[0]
			y2 = y1 + self.score_size[1]
			im_digit = np.mean(self.captured[y1:y2, x1:x2, :], axis=2)

			digit_response = np.matmul(self.number_w, np.reshape(im_digit, [-1, 1])) + self.number_b

			self.score += np.argmax(digit_response)
			self.score *= 10

		self.score /= 10

		x1 = self.airstike_rect[0]
		y1 = self.airstike_rect[1]
		x2 = self.airstike_rect[2]
		y2 = self.airstike_rect[3]
		airstrike = self.captured[y1:y2,x1:x2,:]
		
		if np.mean(airstrike - self.airstrike_orig) > 32.0:
			self.done = True
			self.status = 'DONE'
		else:
			self.done = False

		return self.captured, self.score, self.done