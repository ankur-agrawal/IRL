# if iter == 0:
    #     i=0
    #     with concurrent.futures.ThreadPoolExecutor(max_workers = 4) as executor:
    #     # for actions in action_set:
    #     #     q = r[index_x, index_y, index_z, index_vel_x, index_vel_y, index_speed]
    #     #     p = np.exp(q)
    #     #     action_value.append(q)
    #     #     policy.append(p)
    #         # v = r[index_x, index_y, index_vel_theta, index_speed]
    #         # print v[120,265, 50, 3]
    #         for q, p in executor.map(initial_loop, action_set):
    #             action_value.append(q)
    #             policy.append(p)
    # else:
    #     with concurrent.futures.ThreadPoolExecutor(max_workers = 4) as executor:
    #     # for action in action_set:
    #     #     new_xv, new_yv, new_zv = mdp.get_next_state(xv, yv, zv, action)
    #     #     new_index_x, new_index_y, new_index_z = get_indices(new_xv, new_yv, new_zv)
    #     #     q = r[index_x, index_y, index_z] + 0.9*v[new_index_x,new_index_y, new_index_z]
    #     #     p = np.exp(q)
    #         for q, p in executor.map(rl_loop, action_set):
    #             action_value.append(q)
    #             policy.append(p)
                # print q[150,250,0]



    # print v[120,265, 50, 3]
    # print policy[:,100,100,10]
    # print max(policy[:,100,100,0])
    # print sum(policy[:,200,200,0])
    # print policy.shape
    # print v.shape
    # action_index = np.argmax(policy, axis = 0)
    # plot_index_x, plot_index_y, plot_index_z = get_indices(plot_xv, plot_yv, plot_zv)
    # # print action_index[150,250,0]
    # plot_action_index = action_index[plot_index_x, plot_index_y, plot_index_z]
    # plot_action = action_set[int(plot_action_index.reshape(:,1))]
    # new_plot_xv, new_plot_yv, new_plot_zv = mdp.get_next_state(plot_xv, plot_yv, plot_zv, plot_action)
    #
    # plot_u = new_plot_xv - plot_xv
    # plot_v = new_plot_yv - plot_yv
    # plot_w = 0*(new_plot_zv - plot_zv)
    #
    # # print plot_u.shape
    # # print plot_action_index.dtype
    #
    # print 'ploting figure ...'
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax = fig.gca(projection='3d')
    #
    # value = ax.quiver(plot_xv, plot_yv, plot_zv, new_plot_xv, new_plot_yv, new_plot_zv, length=0.01)
    # value = ax.plot_surface(xv[:,:,0],yv[:,:,0],v[:,:,0])
    # plt.ion()
    # plt.draw()
    # plt.show(value)
        # print v
            # print i

            # q = reward(xv,yv)




# print new_xv
# print zv[200,200,:]
