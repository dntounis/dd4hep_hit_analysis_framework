def plot_full_detector_with_hits(subdetectors_names,subdetectors_rmax,subdetectors_zmax,events_trees):
    Ndet = len(subdetectors_names)
    #hists = []
    rmax = max(subdetectors_rmax)
    zmax = max(subdetectors_zmax)
    print("rmax = ", rmax, " mm")
    print("zmax = ", zmax, " mm")


    combined_histogram = (
        hist.Hist.new.Reg(2000, -zmax, zmax, name="z")
        .Reg(1000, 0, rmax, name="r")
        .Int64()
    )

    for i in range(len(subdetectors_names)):
        print(i, " : ", subdetectors_names[i])
        x_combined, y_combined,z_combined,phi_combined,r_combined= read_hits(events_trees,subdetectors_names[i])
        combined_histogram.fill(z=z_combined, r=r_combined)

    # Plot the final combined histogram
    fig, ax = plt.subplots(figsize=(20, 10))
    #hep.style.use("ATLAS")
    plt.style.use(hep.style.CMS)
    #hep.atlas.label("Internal",loc=1,data='true',lumi=139)
    combined_histogram.plot2d(ax=ax,cmap="plasma", norm=LogNorm())
    ax.set_xlabel('z [mm]', fontsize=16)
    ax.set_ylabel('r [mm]', fontsize=16)
    ax.set_title('Beam-Induced Background Detector Hits')
    plt.show()
    fig.savefig("SiD_hits_550.png")
    #fig.savefig("SiD_hits_550.pdf")
    