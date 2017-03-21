import model as m

modelConfig = {
	"lRate"		: 1,						# learning rate
	"model"		: m.initFirstGrad, 		# trained or untrained model
	"e"		  	: 0.1,						# stop criteria
	"lamda"		: 0
}