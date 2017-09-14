import ggc
import cv2
import agent
import numpy as np
import time

meta = agent.meta_init()
ag = agent.Agent(meta)
game = ggc.GGC()

cv2.namedWindow('Game')

game.start()

episode = 1

while True:

	done = False

	obs_prev = np.zeros([meta.screen_height, meta.screen_width, meta.screen_channel])
	obs_next = np.zeros([meta.screen_height, meta.screen_width, meta.screen_channel])
	action = [0, 0, 0, 0]
	reward = 0
	first = True
	score_prev = 0
	time_before = time.clock()

	print('Episode {} started.'.format(episode))

	while not done:

		screen, score, done = game.capture()
		obs_next = cv2.resize(screen, (meta.screen_width, meta.screen_height), interpolation=cv2.INTER_CUBIC)

		if done:
			reward = -1000
		else:
			reward = (score - score_prev) * meta.ratio_d + score * meta.ratio_p

		if first:
			first = False
			print('First trial')
		else:
			full = ag.feed(obs_prev, action, reward, obs_next)
			if full:
				game.pause()
				for i in range(meta.batch_length):
					loss_value = ag.batch()
					print('batch {}: Loss: {}'.format(i + 1, loss_value))
				ag.decay()
				ag.flush()
				print('Saving status...')
				ag.save()
				game.resume()

		if done:
			break

		action = ag.decide(obs_next)
		print(action)
		game.action(action)

		obs_prev = obs_next
		score_prev = score

		cv2.imshow('Game', obs_next)

		if cv2.waitKey(1) == ord('q'):
			exit()

		if time.clock() - time_before < meta.frame_interval:
			time.sleep(time.clock() - time_before)

		time_before = time.clock()

	print('Episode {} ended.'.format(episode))
	episode += 1
	game.restart()