SH = 0; RE = 1; RA = 2; LA = 3;


def oracle(stack, buffer, heads, labels):

	#for initial shift
	if len(stack)==1 and buffer!= []:
		trans = SH

	#if the head of the top of the stock is the next buffer item --> left arc
	if heads[stack[0]]==buffer[0]:
		trans = (LA, labels[stack[0]])

	#if the head of the next buffer item is the top of the stack --> right arc
	elif heads[buffer[0]]== stack[0]:
		trans = (RA, labels[buffer[0]])

	#add words to the stack if no dependencies
	elif len(buffer)==1 and len(stack)==1:
		trans= SH

	#remove the top of the stack if we just have one word in the buffer
	elif len(buffer)==1 and stack!=[]:
		trans = RE

	else:
		trans = SH

	return trans
