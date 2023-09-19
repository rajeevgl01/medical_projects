import matplotlib.pyplot as plt

# Data
data = [{"train_lr": 9.863013698630137e-05, "train_loss": 0.6849906175103906, "test_loss": 0.32765002533793447, "test_auc_avg": 0.5767120654263147, "test_auc_each_class": [0.5293501031458561, 0.613554112660937, 0.6538900953195517, 0.518850444881073, 0.5304188757055555, 0.47639777744417183, 0.625928360371873, 0.6838645392086202, 0.5366692487682889, 0.6772395207416082, 0.41546607116726636, 0.6292765623865068, 0.5832546475591094, 0.5998085566079876], "epoch": 0, "n_parameters": 30338},
{"train_lr": 0.0002986301369863014, "train_loss": 0.17473046894963473, "test_loss": 0.2342328866571188, "test_auc_avg": 0.6883513812875321, "test_auc_each_class": [0.6890853032402349, 0.7251861915943572, 0.7806308867239397, 0.6244439070559707, 0.7267315903084139, 0.6109214723207683, 0.6738653473168867, 0.7718032308852489, 0.6243157633423353, 0.7438523400381453, 0.588418136823547, 0.6828020650459318, 0.6628175306860208, 0.732045572643651], "epoch": 1, "n_parameters": 30338},
{"train_lr": 0.0004986301369863013, "train_loss": 0.14833007222169067, "test_loss": 0.2135264614224434, "test_auc_avg": 0.7647413146780367, "test_auc_each_class": [0.7655942487632426, 0.8497984104343038, 0.8258721818338638, 0.6882117994721941, 0.8072933018959897, 0.6856932831526246, 0.7001216383509423, 0.8352163595771407, 0.6872646322856766, 0.798930662655684, 0.8246787862748027, 0.7343120876269469, 0.7308284977852597, 0.772562515383844], "epoch": 2, "n_parameters": 30338},
{"train_lr": 0.0006986301369863015, "train_loss": 0.14120891618810288, "test_loss": 0.20571857191622256, "test_auc_avg": 0.7932232184099081, "test_auc_each_class": [0.7734639757773444, 0.8862471601617476, 0.8358119328032246, 0.6963564873944093, 0.8201544564852763, 0.7139207414461622, 0.7173687404908202, 0.855985731891817, 0.723743368522432, 0.8280192632338877, 0.8943609571268586, 0.7809852595263514, 0.768242550325711, 0.8104644325526696], "epoch": 3, "n_parameters": 30338},
{"train_lr": 0.0008986301369863014, "train_loss": 0.13776178170658954, "test_loss": 0.2019231003522873, "test_auc_avg": 0.8055966761146077, "test_auc_each_class": [0.776533610227403, 0.8954008341087463, 0.8376296908975079, 0.7016503707969273, 0.8214993106608827, 0.7228566903632268, 0.7269043093650738, 0.8631632068896141, 0.7394372524399933, 0.8435698330570853, 0.9120198848627644, 0.8065242367886443, 0.7869132346270072, 0.8442510005196321], "epoch": 4, "n_parameters": 30338},
{"train_lr": 0.000999837243155759, "train_loss": 0.1360570460762063, "test_loss": 0.20049403920769693, "test_auc_avg": 0.8115928175556174, "test_auc_each_class": [0.7787307695350876, 0.8993667962108767, 0.8384179471002347, 0.7012533768813933, 0.8226282186110815, 0.7258471798436006, 0.7300246694520087, 0.8655467994652037, 0.7488168495278832, 0.8507574381564086, 0.9169706762198284, 0.8184676887739508, 0.7934974369564782, 0.8719735990446063], "epoch": 5, "n_parameters": 30338},
{"train_lr": 0.000998847534638751, "train_loss": 0.1357374831472766, "test_loss": 0.20045091822743416, "test_auc_avg": 0.814543750234823, "test_auc_each_class": [0.7796771950613252, 0.9016147150485693, 0.8384983640387345, 0.7012227520083603, 0.8230045532460734, 0.7294714306727128, 0.7331355675790802, 0.8653112449347389, 0.7520742477587323, 0.8546743030168914, 0.9191793607138645, 0.8222152784344682, 0.7965215574692022, 0.8870119333047686], "epoch": 6, "n_parameters": 30338},
{"train_lr": 0.0009968632829095094, "train_loss": 0.13501020258113947, "test_loss": 0.19955925956368448, "test_auc_avg": 0.816845759638615, "test_auc_each_class": [0.7803472990987127, 0.9037832078529139, 0.838679843013996, 0.7013718918650932, 0.8243200932373376, 0.7318425388142683, 0.7357852041570742, 0.8660244130711738, 0.7543474934210854, 0.8575744582489344, 0.9190853228980793, 0.8255232166914039, 0.79794512865379, 0.8992105239167496], "epoch": 7, "n_parameters": 30338},
{"train_lr": 0.000993888483986947, "train_loss": 0.13465127132016502, "test_loss": 0.19928784504532815, "test_auc_avg": 0.8179863555008186, "test_auc_each_class": [0.7810124562564997, 0.9046207186650568, 0.8388796908462414, 0.7018918429244132, 0.8239234161896435, 0.7317268304949456, 0.7370915302507491, 0.8664435184990488, 0.7563272668425368, 0.8581064977262943, 0.919100519797434, 0.8268906860507983, 0.7986784618225424, 0.9071155406452553], "epoch": 8, "n_parameters": 30338},
{"train_lr": 0.000989929128720199, "train_loss": 0.1343801696976162, "test_loss": 0.19925599321722984, "test_auc_avg": 0.8189075690338846, "test_auc_each_class": [0.7804616307787633, 0.9048434929559501, 0.8387085933768106, 0.7027100946903944, 0.8242459441834901, 0.7335814919531443, 0.7380363950868323, 0.8657186258179057, 0.7578749060382602, 0.8594630088724371, 0.919534452883059, 0.8274451840492059, 0.7987652066944788, 0.9133169390936522], "epoch": 9, "n_parameters": 30338},
{"train_lr": 0.0009849931907238527, "train_loss": 0.13429133958910425, "test_loss": 0.1998987529426813, "test_auc_avg": 0.8195644284628985, "test_auc_each_class": [0.7807681186673878, 0.9053111435239675, 0.8387071886693549, 0.7033978653514603, 0.8242319947577987, 0.734393390654443, 0.7382322540582993, 0.8652352767577551, 0.7596844275640564, 0.8603249246571366, 0.919418963915728, 0.8274589345762713, 0.7996619043034293, 0.9170756110234928], "epoch": 10, "n_parameters": 30338},
{"train_lr": 0.000979090610320144, "train_loss": 0.13434970641687308, "test_loss": 0.19900620579719544, "test_auc_avg": 0.8202300377393149, "test_auc_each_class": [0.7823900938981977, 0.9060604942995755, 0.8392782022500664, 0.7019452832018969, 0.824389204665398, 0.7353978067182223, 0.7388373158110789, 0.8666684203029187, 0.7596056902615614, 0.8595404167492855, 0.9195464199745655, 0.8285884421566491, 0.8002690468500664, 0.9207036912109251], "epoch": 11, "n_parameters": 30338},
{"train_lr": 0.0009722332745204754, "train_loss": 0.13417317866258424, "test_loss": 0.1990669795125723, "test_auc_avg": 0.8207448200163684, "test_auc_each_class": [0.7823523773575964, 0.9060264355011218, 0.8393077985133561, 0.7044988030855766, 0.8251639555284151, 0.7350771157910012, 0.7382870470806256, 0.8674086048012706, 0.760289130510583, 0.8604539304818986, 0.9196146379969755, 0.8280322995769314, 0.8009863512695394, 0.9229289927342674], "epoch": 12, "n_parameters": 30338},
{"train_lr": 0.0009644349930865555, "train_loss": 0.13436252560007245, "test_loss": 0.19847706653177738, "test_auc_avg": 0.8209726335494522, "test_auc_each_class": [0.7825392518839446, 0.9056813470258602, 0.8394505711335447, 0.7043782349673944, 0.8254248205837582, 0.7365037310021628, 0.7372889722116989, 0.8677018493280642, 0.7617599206163033, 0.8605410444695435, 0.9196939643180538, 0.8297105491211312, 0.8022136239431445, 0.9207289890877267], "epoch": 13, "n_parameters": 30338},
{"train_lr": 0.0009557114707193762, "train_loss": 0.13405762551582023, "test_loss": 0.19888303980231284, "test_auc_avg": 0.8212058533058518, "test_auc_each_class": [0.7818925157032339, 0.9064728612972023, 0.8392273661948477, 0.7051917380726407, 0.8246194481525635, 0.7369838614138173, 0.7390268068475807, 0.8664687103156277, 0.7617332887112811, 0.8602292000565277, 0.9199424728282612, 0.8300637686403013, 0.8025941099570053, 0.9224357980910358], "epoch": 14, "n_parameters": 30338},
{"train_lr": 0.0009460802754320354, "train_loss": 0.13386546567796845, "test_loss": 0.19891904100775717, "test_auc_avg": 0.8213414229883448, "test_auc_each_class": [0.7823996255239473, 0.9061199160330479, 0.8393086290338224, 0.7055516496083241, 0.8249058251842141, 0.738078168603059, 0.7383666282791718, 0.8660754839770147, 0.7620106590174367, 0.8601506309519766, 0.9193996970850967, 0.8299716720869326, 0.8032413601552986, 0.9231999762974848], "epoch": 15, "n_parameters": 30338},
{"train_lr": 0.0009355608031700976, "train_loss": 0.13411069536351994, "test_loss": 0.19898278675973416, "test_auc_avg": 0.8212559354255298, "test_auc_each_class": [0.7817355615992212, 0.9060977186974375, 0.8387254908796343, 0.7028751935740536, 0.8238458727355633, 0.7377886536401511, 0.7383479921757147, 0.8667387512075341, 0.7613590751141355, 0.860694457986015, 0.9199115562860855, 0.8313315580991746, 0.8038008100171382, 0.9243304039455571], "epoch": 16, "n_parameters": 30338},
{"train_lr": 0.0009241742387507385, "train_loss": 0.13397571015847873, "test_loss": 0.19934727557003498, "test_auc_avg": 0.8215349894196523, "test_auc_each_class": [0.7813887744068894, 0.9063027198639418, 0.8396066987279012, 0.7032334802364351, 0.8246973634314043, 0.7392553503981772, 0.7395568924621279, 0.8665850803082222, 0.7608464080464263, 0.8608911217569156, 0.9191377092612107, 0.832530138094579, 0.8033512894681565, 0.9241068254127429], "epoch": 17, "n_parameters": 30338},
{"train_lr": 0.000911943513199346, "train_loss": 0.13411582955350615, "test_loss": 0.19879998140037058, "test_auc_avg": 0.8216512204605421, "test_auc_each_class": [0.7824195087002425, 0.9064289815126324, 0.8394417891486123, 0.7030019756779895, 0.8248886852621907, 0.7405310718859395, 0.739197841665794, 0.866904711196973, 0.761019799240151, 0.8615028258366589, 0.9194436075363029, 0.8312589681074569, 0.8044525861903953, 0.9226247344862479], "epoch": 18, "n_parameters": 30338},
{"train_lr": 0.000898893257569482, "train_loss": 0.13392312490470606, "test_loss": 0.1981428112089634, "test_auc_avg": 0.8217431927226119, "test_auc_each_class": [0.7826940126891289, 0.9070704349300934, 0.8394580714510907, 0.7056647273544894, 0.8247556679527166, 0.7397152022971828, 0.7392576714728386, 0.8662937994176673, 0.7610351018702224, 0.8608180958713973, 0.9191265262849044, 0.8321471333805694, 0.8040612235252379, 0.9223070296190277], "epoch": 19, "n_parameters": 30338},
{"train_lr": 0.000885049753339224, "train_loss": 0.1337806779687127, "test_loss": 0.19952075004577638, "test_auc_avg": 0.8215367362436198, "test_auc_each_class": [0.7813910770290239, 0.906793521261367, 0.8391646208862732, 0.7022261427620569, 0.825251706171888, 0.7385343451822817, 0.7394812327602551, 0.867014912099051, 0.7605262575639997, 0.8616979559105943, 0.9195904797810481, 0.8323690148089979, 0.8044631408355679, 0.9230099003582726], "epoch": 20, "n_parameters": 30338},
{"train_lr": 0.0008704408794837604, "train_loss": 0.1342125919787851, "test_loss": 0.1990950455516577, "test_auc_avg": 0.8217569185671701, "test_auc_each_class": [0.7818523667190722, 0.9068089297211378, 0.8392704097123564, 0.7020964803917512, 0.8247974562580553, 0.7370973979532683, 0.741214498312857, 0.8672424811754784, 0.7609660488968264, 0.8626113162735107, 0.9196957192425492, 0.8313060670888672, 0.8045948413402482, 0.9250428468544027], "epoch": 21, "n_parameters": 30338},
{"train_lr": 0.0008550960563308505, "train_loss": 0.13393546041254312, "test_loss": 0.19852946631610394, "test_auc_avg": 0.8221855912219855, "test_auc_each_class": [0.781827379510924, 0.9062169244920253, 0.839370877055451, 0.7077636941802697, 0.8253256873048761, 0.7372830144530793, 0.7414602934071006, 0.8672688020828944, 0.7600510303906063, 0.8620447686144252, 0.9192531048814944, 0.8329443898534815, 0.804437559237946, 0.9253507516432224], "epoch": 22, "n_parameters": 30338},
{"train_lr": 0.0008390461863121935, "train_loss": 0.13431406074701108, "test_loss": 0.1987942684441805, "test_auc_avg": 0.8218928643954958, "test_auc_each_class": [0.7821965233193513, 0.9065003028487001, 0.8394681812434356, 0.7036944740065952, 0.825549561793725, 0.7371447915873935, 0.7408742635051488, 0.8662891521452358, 0.760791835230176, 0.8619215689281758, 0.9196870566365288, 0.8309110934775448, 0.8042325486723885, 0.9272387481425433], "epoch": 23, "n_parameters": 30338},
{"train_lr": 0.0008223235917300486, "train_loss": 0.1338097127026891, "test_loss": 0.19853998444974422, "test_auc_avg": 0.8222583373420556, "test_auc_each_class": [0.7817287903869503, 0.906761655498648, 0.8398220034688276, 0.7056555323350291, 0.8259289669815776, 0.7397633284251108, 0.7402208845961091, 0.8671218484558824, 0.7608257419661482, 0.861347769073439, 0.9193542370728994, 0.8311220567133865, 0.8052719128834614, 0.9266919949313084], "epoch": 24, "n_parameters": 30338},
{"train_lr": 0.0008049619496644172, "train_loss": 0.13383066202577662, "test_loss": 0.1989626643061638, "test_auc_avg": 0.8218643022494181, "test_auc_each_class": [0.7831017271264863, 0.9064520369926607, 0.8390206076120543, 0.7052613683021967, 0.8253610106571213, 0.7377922261537864, 0.7387183757376642, 0.867850954773121, 0.7606057246664149, 0.860720640384213, 0.9189175782534834, 0.8313607037346158, 0.8048788686307682, 0.9260584084672676], "epoch": 25, "n_parameters": 30338},
{"train_lr": 0.0007869962241518868, "train_loss": 0.1336047047468489, "test_loss": 0.19866803273558617, "test_auc_avg": 0.8218551895440266, "test_auc_each_class": [0.7822922222084201, 0.906584706119672, 0.8400431526130367, 0.7047485717237717, 0.8251857612514653, 0.7384677525000591, 0.740862714877331, 0.8676799629658205, 0.7585007615983452, 0.8614584362644839, 0.9190129042585258, 0.8306852376442834, 0.8052508214823452, 0.9251996481088127], "epoch": 26, "n_parameters": 30338},
{"train_lr": 0.0007684625957727136, "train_loss": 0.13395127368299928, "test_loss": 0.19942036099731922, "test_auc_avg": 0.8217176465780766, "test_auc_each_class": [0.7828179306565823, 0.9058014681744937, 0.8399108640320698, 0.7023950582085878, 0.8248507951196633, 0.7396303101206562, 0.7401620261689749, 0.8667695557510283, 0.7600264141234587, 0.8602934838693422, 0.9187384639384859, 0.8317340693748352, 0.8046328022844116, 0.9262838102704821], "epoch": 27, "n_parameters": 30338},
{"train_lr": 0.000749398388787949, "train_loss": 0.13385624178264238, "test_loss": 0.19908624999225139, "test_auc_avg": 0.8221887744152593, "test_auc_each_class": [0.7825920618734928, 0.9067114063755097, 0.8395081692658949, 0.7048264564776467, 0.8248936509218951, 0.7378686368231214, 0.7401544350148639, 0.8673085248076394, 0.7608551888137196, 0.8613870536257144, 0.9197714423675888, 0.831033477736709, 0.8052157049256097, 0.928516632784225], "epoch": 28, "n_parameters": 30338},
{"train_lr": 0.0007298419959733454, "train_loss": 0.13367732655103892, "test_loss": 0.19886399790644646, "test_auc_avg": 0.8219525119569163, "test_auc_each_class": [0.7820083505787432, 0.9065583896908556, 0.8390881104699675, 0.7061061596654417, 0.8252383684578992, 0.7366386640711199, 0.7402557463417653, 0.8669432230127039, 0.7613334278597991, 0.8620133935565008, 0.9197418140146704, 0.8304762844522654, 0.8046653785706142, 0.9262678566544811], "epoch": 29, "n_parameters": 30338},
{"train_lr": 0.0007098328013014156, "train_loss": 0.1338807080859599, "test_loss": 0.19839940257370473, "test_auc_avg": 0.8227161642479538, "test_auc_each_class": [0.782816769096455, 0.9060240326967517, 0.8395711811612856, 0.7070619294553404, 0.8258569049441273, 0.7391191194016045, 0.7422778714979507, 0.8661771839705428, 0.7610679892031316, 0.8627768679059669, 0.9195430034726222, 0.8322899835404821, 0.8052354188560847, 0.9282080442690053], "epoch": 30, "n_parameters": 30338},
{"train_lr": 0.0006894111006273525, "train_loss": 0.13369915062842302, "test_loss": 0.19926158301532268, "test_auc_avg": 0.8220398291798245, "test_auc_each_class": [0.7827548101127283, 0.9072438373121422, 0.839798010655352, 0.7032630386551565, 0.8251014170025723, 0.7403597711422064, 0.7410330301548704, 0.8662975794174268, 0.7604200542994308, 0.8603609227159145, 0.9191569014142041, 0.8308499241893699, 0.8046207628332231, 0.927297548612947], "epoch": 31, "n_parameters": 30338},
{"train_lr": 0.00066861802053854, "train_loss": 0.1337122446921182, "test_loss": 0.19911444678902626, "test_auc_avg": 0.821880212723556, "test_auc_each_class": [0.7823309226587761, 0.9069246075886741, 0.8389449789829106, 0.7042040796193628, 0.8252219122136613, 0.738129995752307, 0.7409006706478852, 0.8667492648432287, 0.760424016070426, 0.8615773415992296, 0.9192254928248045, 0.8300576928260165, 0.8049887800543968, 0.9266432224481052], "epoch": 32, "n_parameters": 30338},
{"train_lr": 0.0006474954355310797, "train_loss": 0.1337314700988466, "test_loss": 0.19875965133309365, "test_auc_avg": 0.8225542911934328, "test_auc_each_class": [0.7825328838013649, 0.9071892783970382, 0.839760242480806, 0.7055256810259674, 0.8254361672361262, 0.73926457724994, 0.7420348826123356, 0.8670592493689567, 0.7615712962972616, 0.8616020998502455, 0.9195268917722008, 0.8299747328354821, 0.8055061007816913, 0.9287759929986417], "epoch": 33, "n_parameters": 30338},
{"train_lr": 0.0006260858836801265, "train_loss": 0.13378644002320833, "test_loss": 0.19874607935547828, "test_auc_avg": 0.8221670648336511, "test_auc_each_class": [0.7822855944829884, 0.9063197301932926, 0.8396458869899113, 0.7078626854925131, 0.8254651575730964, 0.7398436714295984, 0.74081040427033, 0.8666135203064124, 0.7598246534058546, 0.862074807164994, 0.9188685337146574, 0.8297043362584039, 0.8051850964546088, 0.9258348299344534], "epoch": 34, "n_parameters": 30338},
{"train_lr": 0.0006044324809738693, "train_loss": 0.13341432860861085, "test_loss": 0.1990435615181923, "test_auc_avg": 0.8220332024107703, "test_auc_each_class": [0.7819961063684611, 0.9067954473188384, 0.8396056580139831, 0.704683203112824, 0.8251842859467705, 0.7401000442606168, 0.7403867387214698, 0.8675135775218632, 0.7613290027237755, 0.862577947409531, 0.9189915837928466, 0.829840059899306, 0.8053687115021554, 0.924092467158342], "epoch": 35, "n_parameters": 30338},
{"train_lr": 0.0005825788344836669, "train_loss": 0.13349345206499916, "test_loss": 0.19828520357608795, "test_auc_avg": 0.8219005615664623, "test_auc_each_class": [0.782145155030902, 0.9066207863249767, 0.8393795206203056, 0.70699734598989, 0.8253429231816762, 0.7381989015728545, 0.740981978744049, 0.8670040548270146, 0.7596071730296532, 0.8612931694614643, 0.9187266835410747, 0.8301284098223534, 0.8048736270866399, 0.9253081326976198], "epoch": 36, "n_parameters": 30338},
{"train_lr": 0.0005605689545452096, "train_loss": 0.13341453395885963, "test_loss": 0.198596895262599, "test_auc_avg": 0.8223487726480914, "test_auc_each_class": [0.7829194715098204, 0.9071018048760375, 0.8397646462899462, 0.7068020756862121, 0.8257330153328056, 0.7395502883853804, 0.7413630115079737, 0.8667711675691075, 0.7619569550106151, 0.8617272276126802, 0.9194460718983605, 0.8305623965569777, 0.805573561064651, 0.923611123772711], "epoch": 37, "n_parameters": 30338},
{"train_lr": 0.0005384471661275613, "train_loss": 0.13368436311409898, "test_loss": 0.19855340041220187, "test_auc_avg": 0.8221180168755168, "test_auc_each_class": [0.7830674542700269, 0.9071034830251214, 0.8393584705151496, 0.7062044120172171, 0.8250827418041187, 0.740802094593016, 0.7410098609451671, 0.865840633082869, 0.7596523047834446, 0.8618195999899214, 0.9184058870771803, 0.8304074861341239, 0.8056131499286632, 0.9252846580912182], "epoch": 38, "n_parameters": 30338},
{"train_lr": 0.0005162580195685681, "train_loss": 0.13374433750669434, "test_loss": 0.198836600035429, "test_auc_avg": 0.8220945421057483, "test_auc_each_class": [0.7816703502471383, 0.907106991882297, 0.8392694305184731, 0.7055346115311691, 0.8259683804061881, 0.7392505956138425, 0.7406067742595837, 0.8660765312496752, 0.7610824230237743, 0.8614298875909675, 0.9184858108193633, 0.8315065689602638, 0.8052501595808683, 0.9260850737968693], "epoch": 39, "n_parameters": 30338},
{"train_lr": 0.0004940462008564113, "train_loss": 0.133541967337058, "test_loss": 0.19890633843839167, "test_auc_avg": 0.8220391842324365, "test_auc_each_class": [0.7822468462036859, 0.9072413200885162, 0.83966250252591, 0.7032981602705831, 0.8259178122387631, 0.7410975080586526, 0.7392004320122207, 0.8669628266478201, 0.7601178823801161, 0.8619413974389452, 0.9190821304290503, 0.8308873840969901, 0.8054686944036301, 0.925423682459227], "epoch": 40, "n_parameters": 30338},
{"train_lr": 0.00047185644163796913, "train_loss": 0.13392160919635263, "test_loss": 0.19885489717125893, "test_auc_avg": 0.821961812935992, "test_auc_each_class": [0.7820729333218162, 0.9070674600294446, 0.8394667201426149, 0.7062964671777945, 0.824950839968926, 0.7387337120043578, 0.7401569174301892, 0.8667831293865281, 0.7614059560709102, 0.8608461625258674, 0.9187330124709042, 0.8301971624576806, 0.8057428289504542, 0.9250120791664007], "epoch": 41, "n_parameters": 30338},
{"train_lr": 0.00044973342913523604, "train_loss": 0.13362241806845143, "test_loss": 0.19857826411724092, "test_auc_avg": 0.8220685875780822, "test_auc_each_class": [0.78253759836894, 0.9069414081493894, 0.8392172717825114, 0.7056743674296858, 0.8249479853143133, 0.7404764946722384, 0.740602888739944, 0.866502623949833, 0.7612409749524559, 0.8615816797706465, 0.918811685362649, 0.8298488309996267, 0.8049581000264797, 0.9256183165744396], "epoch": 42, "n_parameters": 30338},
{"train_lr": 0.00042772171615120425, "train_loss": 0.1335565292927092, "test_loss": 0.1988740488141775, "test_auc_avg": 0.8220897262882317, "test_auc_each_class": [0.7824270930046026, 0.9070272416610579, 0.8395408261509036, 0.70399709931145, 0.8249060410824621, 0.741399963745412, 0.7406722524609191, 0.8657966476311225, 0.7626795264363072, 0.862055592133011, 0.9188594043733989, 0.8305044707486089, 0.804531996478398, 0.9248580128175908], "epoch": 43, "n_parameters": 30338},
{"train_lr": 0.00040586563134644647, "train_loss": 0.13379795612027384, "test_loss": 0.19863952316343783, "test_auc_avg": 0.8219761345149372, "test_auc_each_class": [0.7825818469770801, 0.9070537869283857, 0.8396655375142814, 0.7050758766288032, 0.8244435270634691, 0.7385765599567123, 0.740505139139379, 0.8666375421230655, 0.7626414609992027, 0.8614224820256192, 0.9182427164379184, 0.8301460433886233, 0.8053928082937615, 0.9252805557328181], "epoch": 44, "n_parameters": 30338},
{"train_lr": 0.00038420918996709457, "train_loss": 0.13373044401101053, "test_loss": 0.19878433026373388, "test_auc_avg": 0.8220234930130018, "test_auc_each_class": [0.7822783791447865, 0.9069772976559346, 0.8394263117332528, 0.7048097458943356, 0.8252682463765557, 0.7394567604924404, 0.7407873429917278, 0.8668523066548532, 0.7614930107756694, 0.86160050042341, 0.918748078684392, 0.8301458149745523, 0.8050559541096907, 0.9254291522704274], "epoch": 45, "n_parameters": 30338},
{"train_lr": 0.00036279600520399053, "train_loss": 0.13376747344760861, "test_loss": 0.19880565360188485, "test_auc_avg": 0.8221858146724763, "test_auc_each_class": [0.7816518472775826, 0.9067785132689913, 0.8392851847739882, 0.7054353137182771, 0.825078495805241, 0.7398598376963363, 0.7428159080369455, 0.8674113211647341, 0.7613345283517423, 0.8622426812528552, 0.9188645011221995, 0.8311234271978116, 0.8050392097912474, 0.9236806359567156], "epoch": 46, "n_parameters": 30338},
{"train_lr": 0.000341669200361524, "train_loss": 0.13370812036199112, "test_loss": 0.19828703723847865, "test_auc_avg": 0.8224766689007647, "test_auc_each_class": [0.7824619261430056, 0.9066787587478764, 0.8396685519959745, 0.7079512305908673, 0.8253112581053001, 0.7398349457578058, 0.7422411029695084, 0.8660178021625036, 0.7619583566898268, 0.8614938208444755, 0.9187337779166949, 0.8312282692563341, 0.8056297690225028, 0.9254637944080296], "epoch": 47, "n_parameters": 30338},
{"train_lr": 0.0003208713220130369, "train_loss": 0.13351654712978292, "test_loss": 0.1985424242913723, "test_auc_avg": 0.8220766584608681, "test_auc_each_class": [0.78221319512353, 0.9071684922322483, 0.8395099892335839, 0.704128436938726, 0.8246066861672361, 0.74000464015548, 0.7419460553161284, 0.8659836185283152, 0.7607766947778647, 0.8613385230717321, 0.9187191597690355, 0.8315398717317943, 0.8053128255504275, 0.9258250298560528], "epoch": 48, "n_parameters": 30338},
{"train_lr": 0.0003004442543176872, "train_loss": 0.13377838540975362, "test_loss": 0.19851350225508213, "test_auc_avg": 0.8223610054116212, "test_auc_each_class": [0.7820659571283463, 0.9068871734221766, 0.8396131737115378, 0.7048584711001966, 0.8256690494796565, 0.7404072675824793, 0.7419092867876862, 0.8658449939916822, 0.761604623826948, 0.8618517418963287, 0.9191344794533628, 0.8312893928616948, 0.8052376371204939, 0.9266808274001077], "epoch": 49, "n_parameters": 30338},
{"train_lr": 0.0002804291346713253, "train_loss": 0.13354672705882217, "test_loss": 0.19879110679030418, "test_auc_avg": 0.8220977453355914, "test_auc_each_class": [0.7826542326711252, 0.9069173419659357, 0.8395812960803002, 0.705701133753457, 0.825217594248701, 0.7392431678696656, 0.7400682700191505, 0.8661389421547945, 0.7613940128372964, 0.8612408265750247, 0.9189897355213035, 0.8302695697181417, 0.8054485690208856, 0.9265037422624963], "epoch": 50, "n_parameters": 30338},
{"train_lr": 0.00026086627086125384, "train_loss": 0.13340701532792554, "test_loss": 0.19880068860948086, "test_auc_avg": 0.8222321670193925, "test_auc_each_class": [0.7824990345727166, 0.9070342784452848, 0.8396098413763328, 0.7058976804433978, 0.8252629448751321, 0.7390892156633699, 0.7404645210683307, 0.8664375375903386, 0.7608663906632873, 0.861673789228408, 0.919182609191122, 0.8310484160169429, 0.8055713964679292, 0.9266126826689032], "epoch": 51, "n_parameters": 30338},
{"train_lr": 0.0002417950598917094, "train_loss": 0.13360793273640822, "test_loss": 0.19847071081399917, "test_auc_avg": 0.8223441254602977, "test_auc_each_class": [0.7826462725679005, 0.9068976427840754, 0.8395363351883811, 0.705194009536352, 0.8255009726935735, 0.739184324200832, 0.7402035436658654, 0.8668352230195768, 0.7613037956662163, 0.8623760033390774, 0.9191067740496253, 0.8308284532667096, 0.8053003925361986, 0.9279040139297858], "epoch": 52, "n_parameters": 30338},
{"train_lr": 0.00022325390864353777, "train_loss": 0.13375523472077225, "test_loss": 0.19867650873959064, "test_auc_avg": 0.822344187300467, "test_auc_each_class": [0.7822194743809237, 0.9069111061164988, 0.8396432416284255, 0.7058641711053559, 0.8249545462221837, 0.7396548422376694, 0.7409879509316433, 0.8669118866510618, 0.7606790753504517, 0.8621349280860447, 0.9191468759412883, 0.830737727197766, 0.805589554035472, 0.9273832423217525], "epoch": 53, "n_parameters": 30338},
{"train_lr": 0.00020528015652784715, "train_loss": 0.1335509010424761, "test_loss": 0.19856909938156606, "test_auc_avg": 0.8222204453570087, "test_auc_each_class": [0.7824054059937569, 0.9070663349067635, 0.8398063107333471, 0.7047119553928528, 0.8251138071631393, 0.7398363593423377, 0.740979676213892, 0.8664358684995356, 0.7610899990419928, 0.8622201797273743, 0.918962982257452, 0.8302808533732419, 0.8053520924083157, 0.9268244099441169], "epoch": 54, "n_parameters": 30338},
{"train_lr": 0.0001879100002894047, "train_loss": 0.13358201334023312, "test_loss": 0.1986188091337681, "test_auc_avg": 0.8222849756645966, "test_auc_each_class": [0.7824728926371478, 0.9066481897367223, 0.8396074010816288, 0.7053366163107654, 0.8254831730824589, 0.7394436655593672, 0.7416946478046275, 0.867287333899897, 0.7604798168040023, 0.8626973128533665, 0.9190518486467983, 0.8301887111370589, 0.8056912006352537, 0.925906849115258], "epoch": 55, "n_parameters": 30338},
{"train_lr": 0.00017117842111120853, "train_loss": 0.13369098489415154, "test_loss": 0.19858371429145336, "test_auc_avg": 0.8223004149297267, "test_auc_each_class": [0.7825405364327912, 0.9070219211656667, 0.839823059562754, 0.7055217888875932, 0.82515537957023, 0.7406932614348257, 0.740911283872827, 0.8667290966626937, 0.7609411777789131, 0.8623559119088283, 0.9188902275685271, 0.8296901289031967, 0.8054927911952362, 0.9264392440720921], "epoch": 56, "n_parameters": 30338},
{"train_lr": 0.00015511911416704087, "train_loss": 0.13335462881583873, "test_loss": 0.1987679846584797, "test_auc_avg": 0.8223220308500789, "test_auc_each_class": [0.7831062845418089, 0.9068841413118999, 0.8394565078168792, 0.7049070115659406, 0.8254106432654715, 0.7404105573792081, 0.7414839662952757, 0.8667374339348906, 0.761079862931991, 0.861928777303914, 0.9190946762722522, 0.8301011828651073, 0.8054428444675716, 0.9264645419488939], "epoch": 57, "n_parameters": 30338},
{"train_lr": 0.0001397644207638726, "train_loss": 0.13320073365450721, "test_loss": 0.19848940394818781, "test_auc_avg": 0.8222540776598629, "test_auc_each_class": [0.7830127516207418, 0.9067417465481521, 0.839508948519666, 0.7045824231803568, 0.8253844596057257, 0.7397739431598688, 0.740989677829261, 0.8667410012073908, 0.7615778529124171, 0.862020601932239, 0.9190599885093517, 0.830627677298428, 0.8056127563656227, 0.925923258548859], "epoch": 58, "n_parameters": 30338},
{"train_lr": 0.00012514526321077477, "train_loss": 0.1335915375041635, "test_loss": 0.1986192611604929, "test_auc_avg": 0.8222026946559187, "test_auc_each_class": [0.7829487838212659, 0.9068178734929601, 0.8396903864815712, 0.704306177924483, 0.825200382360595, 0.7396123061940267, 0.7408091090971167, 0.8666450366680432, 0.7611116266047055, 0.8624329692263704, 0.9189051630961484, 0.8305600667334551, 0.8056432575012489, 0.9261545859808739], "epoch": 59, "n_parameters": 30338},
{"train_lr": 0.00011129108254550323, "train_loss": 0.13361225090920925, "test_loss": 0.19856866911053658, "test_auc_avg": 0.8221868397221416, "test_auc_each_class": [0.7829144426377401, 0.9069249889861931, 0.8398008918436368, 0.7040377631303871, 0.8254766721463243, 0.739857653065696, 0.7408737238496433, 0.8665181857670247, 0.7612940186641116, 0.8622908831574877, 0.9188523660060072, 0.8308487364362014, 0.8056539910387126, 0.9252714393808172], "epoch": 60, "n_parameters": 30338},
{"train_lr": 9.822977924416718e-05, "train_loss": 0.13353000742013324, "test_loss": 0.19837895259261132, "test_auc_avg": 0.8223435806424702, "test_auc_each_class": [0.7829043438973401, 0.9070760414736238, 0.8397846556811843, 0.7050086900071586, 0.8253602909962946, 0.7399659207900735, 0.74081594473352, 0.8664495075895768, 0.7613311457870331, 0.8622948269496848, 0.9188565479537412, 0.8305663709618106, 0.8057537592694383, 0.926642082904105], "epoch": 61, "n_parameters": 30338},
{"train_lr": 8.598765703338288e-05, "train_loss": 0.13313847570999027, "test_loss": 0.19860026761889457, "test_auc_avg": 0.8224090985886746, "test_auc_each_class": [0.7826374788744668, 0.9070319137806666, 0.8397138717547618, 0.7047896973930012, 0.8254920249106278, 0.7404296022180837, 0.740716863982708, 0.8665251730393071, 0.7613115686145722, 0.8622884949722127, 0.9190229297314416, 0.8310958804608666, 0.8058513450142128, 0.9268205354945165], "epoch": 62, "n_parameters": 30338},
{"train_lr": 7.458936991806904e-05, "train_loss": 0.1333363552462973, "test_loss": 0.19851464085280895, "test_auc_avg": 0.8223990617803942, "test_auc_each_class": [0.7825460709251619, 0.9071075830484516, 0.8399204662841295, 0.7048849939029044, 0.8254091439720824, 0.7403767084550527, 0.7408115195583747, 0.8665977457619616, 0.7612260430144073, 0.8621703783959064, 0.9189329618469333, 0.8312845048005786, 0.8057119879194751, 0.9266067570401029], "epoch": 63, "n_parameters": 30338},
{"train_lr": 6.405787253156187e-05, "train_loss": 0.13322854611053042, "test_loss": 0.19858018703758717, "test_auc_avg": 0.8223252500659639, "test_auc_each_class": [0.7825898344111313, 0.9070455868817254, 0.8396005057110894, 0.704058092940536, 0.8254496608766275, 0.7408052944343656, 0.7408105122014311, 0.8667486593887217, 0.760945591330811, 0.8621670261725386, 0.9188897795026985, 0.8312956514072363, 0.8056971219700878, 0.9264501836944927], "epoch": 64, "n_parameters": 30338},
{"train_lr": 5.441437390803841e-05, "train_loss": 0.13346877477246605, "test_loss": 0.1987082725018263, "test_auc_avg": 0.8223468942201223, "test_auc_each_class": [0.7825010365557594, 0.9069950707803238, 0.8396426418080885, 0.7043382765198133, 0.8255352645319671, 0.7407875475231056, 0.7408359839412912, 0.8668135166573216, 0.760905139563809, 0.8620933868082343, 0.9188642397504663, 0.8311797541076844, 0.8056988751145403, 0.9266657854193066], "epoch": 65, "n_parameters": 30338},
{"train_lr": 4.567829477034436e-05, "train_loss": 0.13382265402948204, "test_loss": 0.19853682197630407, "test_auc_avg": 0.8223558516590457, "test_auc_each_class": [0.7825995573526667, 0.9070519371504182, 0.839770870066775, 0.7045949771110354, 0.8254233692677575, 0.7404498164768905, 0.7407516537742966, 0.8666844648473523, 0.7612543893700355, 0.8620007296015565, 0.9188540649222742, 0.8310980732359468, 0.8057653693791282, 0.9266826506705077], "epoch": 66, "n_parameters": 30338},
{"train_lr": 3.786722841924262e-05, "train_loss": 0.13357299637712844, "test_loss": 0.19852177053689957, "test_auc_avg": 0.8223108057387724, "test_auc_each_class": [0.7826343836583631, 0.9069513435547615, 0.8397359113071403, 0.704635044723231, 0.8253899050393145, 0.7401890615338231, 0.7406840529279728, 0.8667172657543558, 0.7611443749281059, 0.8621032462887271, 0.9188060472009719, 0.8310153873422972, 0.8057147965284449, 0.9266304595553043], "epoch": 67, "n_parameters": 30338},
{"train_lr": 3.099690530284704e-05, "train_loss": 0.133377869422101, "test_loss": 0.19848776005208493, "test_auc_avg": 0.8223176565261934, "test_auc_each_class": [0.7826592410450854, 0.9069678199275857, 0.839767958118473, 0.7046724797887595, 0.8254283948991977, 0.7402211627534663, 0.7405927791934741, 0.8666793675749493, 0.761241252971473, 0.8621156035042785, 0.9188022386414286, 0.8311015908126379, 0.8056224523277983, 0.9265748498081008], "epoch": 68, "n_parameters": 30338},
{"train_lr": 2.5081161337593394e-05, "train_loss": 0.13316721571226642, "test_loss": 0.19845617286860942, "test_auc_avg": 0.822337453156121, "test_auc_each_class": [0.7827698693981423, 0.9070296635353041, 0.8398271916584084, 0.704739288532892, 0.8254272914192634, 0.7402237329071607, 0.7407348884765922, 0.8666202539423474, 0.7611718640584313, 0.8621565532132593, 0.918853411492941, 0.8310098140389683, 0.8057048680062908, 0.9264556535056931], "epoch": 69, "n_parameters": 30338},
{"train_lr": 2.0131910044544938e-05, "train_loss": 0.13314978469622463, "test_loss": 0.19849582523107528, "test_auc_avg": 0.8223463371405545, "test_auc_each_class": [0.7827278004234175, 0.9070073136406861, 0.8398524917726182, 0.7047966881269745, 0.8254291145600243, 0.7402464787673564, 0.7407330536478733, 0.866692916664996, 0.7611919741006752, 0.8621384774990223, 0.918928518527466, 0.8310940531482995, 0.8057232939122703, 0.9262865451760824], "epoch": 70, "n_parameters": 30338},
{"train_lr": 1.615911855714649e-05, "train_loss": 0.13313825653023917, "test_loss": 0.19852364040911197, "test_auc_avg": 0.8223483445570318, "test_auc_each_class": [0.7826932815895195, 0.907038740796258, 0.839854696240523, 0.7048381286941305, 0.8254087601529749, 0.7403064019007417, 0.7407310029569523, 0.8667132075727959, 0.7611174070834381, 0.8621086799579767, 0.9189185677321884, 0.8310979361875042, 0.8057281418933581, 0.9263218710400846], "epoch": 71, "n_parameters": 30338},
{"train_lr": 1.3170787548744136e-05, "train_loss": 0.13344922141261298, "test_loss": 0.19849220737814904, "test_auc_avg": 0.8223603870400302, "test_auc_each_class": [0.7827114019275039, 0.9070265932852756, 0.8398911161009783, 0.7050330673053256, 0.8254324849715629, 0.740284439937423, 0.740670453609234, 0.8667260612083416, 0.7611157389693348, 0.8620594921052949, 0.9189142924374067, 0.8310821299338012, 0.8057514336696544, 0.9263467130992862], "epoch": 72, "n_parameters": 30338},
{"train_lr": 1.1172935120293863e-05, "train_loss": 0.13334951350746088, "test_loss": 0.1985104063898325, "test_auc_avg": 0.822363048418576, "test_auc_each_class": [0.7827168885909283, 0.9070247244374318, 0.8398418590599797, 0.7051039655238666, 0.8254280950405197, 0.7403025338194318, 0.7406867871825341, 0.8667050012096817, 0.7610580963597695, 0.862048997236059, 0.9189069180206437, 0.8310811705947033, 0.8057666574036239, 0.9264109833808902], "epoch": 73, "n_parameters": 30338},
{"train_lr": 1.0169584680707104e-05, "train_loss": 0.13325618477920964, "test_loss": 0.1984798440337181, "test_auc_avg": 0.8223403399955913, "test_auc_each_class": [0.7827270283275681, 0.9070300258629472, 0.8398415309531286, 0.7050213950888419, 0.8254001482117483, 0.7402368535417708, 0.7406740153355704, 0.8666762012115145, 0.7610765730402873, 0.8620399922438754, 0.9189207707225127, 0.8310317874725847, 0.8057481062730407, 0.9263403316528857], "epoch": 74, "n_parameters": 30338}]

# Extract train_loss and test_loss values
# train_loss_values = [item["train_loss"] for item in data]
# test_loss_values = [item["test_loss"] for item in data]
test_auc_values = [item["test_auc_avg"] for item in data]

# Create a figure and axis
plt.figure(figsize=(10, 6))
plt.title("test auc Over Time")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# # Plot train_loss as a blue line
# plt.plot(range(1, len(data) + 1), train_loss_values, label="Train Loss", marker='o', linestyle='-')

# # Plot test_loss as a red line
# plt.plot(range(1, len(data) + 1), test_loss_values, label="Test Loss", marker='o', linestyle='-')

# Plot test_loss as a red line
plt.plot(range(1, len(data) + 1), test_auc_values, label="Test auc", marker='o', linestyle='-')

# Add legend
plt.legend()

# Display the plot
plt.grid(True)
plt.tight_layout()
plt.savefig("auc_plot.png")
plt.show()
